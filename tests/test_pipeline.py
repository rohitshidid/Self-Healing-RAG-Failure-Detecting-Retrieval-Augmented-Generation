"""Unit tests for failure detection and healing routing (no models, no API)."""
import numpy as np
import faiss
import pytest

from src.ingest import chunk_text
from src.generation import is_non_answer
from src.pipeline import SelfHealingRAG
from src.retrieval import Retriever, avg_similarity
from tests.fakes import FakeEmbedder, FakeLLM, FakeVerifier

DIM = 64


def make_retriever(vector_map, corpus):
    """Build a tiny FAISS index whose chunk embeddings come from FakeEmbedder."""
    embedder = FakeEmbedder(vector_map, dim=DIM)
    embeddings = embedder.encode(corpus)
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)
    metadata = [{"source": "doc.txt", "chunk_id": i, "text": t} for i, t in enumerate(corpus)]
    return Retriever(embedder, index, metadata)


# Corpus chunks all embed to (approximately) basis vector e0 via the "raft" key.
E0 = np.eye(DIM)[0]
CORPUS = ["raft leader election uses randomized timeouts",
          "raft log replication copies entries to followers",
          "raft terms act as a logical clock"]


def test_chunking_covers_text_with_overlap():
    text = ("Sentence one is here. " * 60).strip()
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Every chunk content must come from the source text.
    assert all(c[:30] in text for c in chunks)


def test_is_non_answer():
    assert is_non_answer("I don't know.")
    assert is_non_answer("")
    assert is_non_answer("I do not know based on the context provided.")
    assert not is_non_answer("Raft elects a leader using randomized timeouts.")


def test_no_healing_when_everything_ok():
    retriever = make_retriever({"raft": E0}, CORPUS)
    llm = FakeLLM(answers=["Raft elects a leader using randomized election timeouts."])
    pipeline = SelfHealingRAG(retriever, llm, FakeVerifier())

    result = pipeline.answer("how does raft elect a leader")

    assert result.healing_steps == 0
    assert not result.healed
    assert result.log[-1].message == "Answer returned (no healing needed)"
    assert not any(e.kind == "failure" for e in result.log)


def test_low_similarity_triggers_query_rewrite():
    # Query has no "raft" keyword -> random embedding -> low similarity.
    # Rewritten query contains "raft" -> high similarity.
    retriever = make_retriever({"raft": E0}, CORPUS)
    llm = FakeLLM(rewrite="raft leader election details",
                  answers=["Raft elects a leader using randomized election timeouts."])
    pipeline = SelfHealingRAG(retriever, llm, FakeVerifier())

    result = pipeline.answer("why cluster keep picking new boss")

    assert result.healing_steps == 1
    assert ("rewrite", "why cluster keep picking new boss") in llm.calls
    assert any(e.kind == "failure" and "low retrieval confidence" in e.message for e in result.log)
    # Healed retrieval should now be high-similarity.
    assert avg_similarity(result.chunks) > 0.9


def test_irrelevant_chunks_trigger_hyde():
    retriever = make_retriever({"raft": E0, "hypothetical": E0}, CORPUS)
    llm = FakeLLM(hyde="hypothetical raft answer",
                  answers=["Raft elects a leader using randomized election timeouts."])
    # Similarity fine ("raft" in query) but relevance below threshold -> HyDE path.
    verifier = FakeVerifier(relevance=-11.0)
    pipeline = SelfHealingRAG(retriever, llm, verifier)

    result = pipeline.answer("raft question that retrieves wrong chunks")

    assert result.healing_steps == 1
    assert any(call[0] == "hyde" for call in llm.calls)
    assert any("HyDE" in e.message for e in result.log)


def test_empty_answer_triggers_retrieval_expansion():
    retriever = make_retriever({"raft": E0}, CORPUS)
    # First generation refuses, second (after expansion) answers.
    llm = FakeLLM(answers=["I don't know.",
                           "Raft elects a leader using randomized election timeouts."])
    pipeline = SelfHealingRAG(retriever, llm, FakeVerifier())

    result = pipeline.answer("raft leader election")

    assert result.healing_steps == 1
    assert any("expanding retrieval" in e.message for e in result.log)
    assert not is_non_answer(result.answer)
    # Second generate call must have received more chunks (expansion), capped by corpus size.
    gen_calls = [c for c in llm.calls if c[0] == "generate"]
    assert gen_calls[1][2] >= gen_calls[0][2]


def test_ungrounded_claim_triggers_reretrieval():
    retriever = make_retriever({"raft": E0, "refined": E0}, CORPUS)
    llm = FakeLLM(refined="refined raft evidence query",
                  answers=["Raft leaders are elected by a council of elders every full moon.",
                           "Raft elects a leader using randomized election timeouts."])
    # First groundedness check fails, second passes.
    verifier = FakeVerifier(groundedness_sequence=[False, True])
    pipeline = SelfHealingRAG(retriever, llm, verifier)

    result = pipeline.answer("raft leader election")

    assert result.healing_steps == 1
    assert any(call[0] == "refine" for call in llm.calls)
    assert any("ungrounded claim" in e.message for e in result.log)
    assert result.groundedness["all_grounded"]


def test_healing_attempts_are_capped():
    retriever = make_retriever({"raft": E0}, CORPUS)
    # Refuses forever + never grounded: pipeline must still terminate.
    llm = FakeLLM(answers=["I don't know."], rewrite="still no match at all")
    verifier = FakeVerifier(relevance=-11.0, groundedness_sequence=[False])
    pipeline = SelfHealingRAG(retriever, llm, verifier)

    result = pipeline.answer("completely unrelated question about cooking")

    from src import config
    assert result.healing_steps <= config.MAX_HEALING_ATTEMPTS
    assert result.log[-1].kind == "success"  # always returns
