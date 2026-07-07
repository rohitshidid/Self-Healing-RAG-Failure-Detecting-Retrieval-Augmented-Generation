"""Integration smoke test with REAL embedding + NLI models.

Uses the real corpus, real FAISS index, real MiniLM embedder, real DeBERTa NLI.
LLM is faked unless GROQ_API_KEY is set, so it runs offline.

Run:  python -m tests.smoke_test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.ingest import build_index
from src.pipeline import SelfHealingRAG
from src.retrieval import Retriever
from src.verification import Verifier


def main():
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("Loading real models (downloads on first run)...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    nli = CrossEncoder(config.NLI_MODEL)
    relevance = CrossEncoder(config.RELEVANCE_MODEL)

    print("Building index from data/docs...")
    index, metadata = build_index(embedder)
    print(f"  {index.ntotal} chunks indexed, dim={index.d}")
    retriever = Retriever(embedder, index, metadata)
    verifier = Verifier(nli, relevance)

    if os.environ.get("GROQ_API_KEY"):
        from src.generation import LLMClient
        llm = LLMClient()
        print("Using REAL Groq LLM.")
    else:
        from tests.fakes import FakeLLM
        llm = FakeLLM(
            rewrite="Raft consensus leader election randomized timeout mechanism",
            refined="follower receives no communication election timeout candidate RequestVote",
            answers=["Raft elects a leader using randomized election timeouts. "
                     "A follower that hears nothing from a leader becomes a candidate and requests votes."])
        print("No GROQ_API_KEY — using FakeLLM (retrieval/verification still real).")

    pipeline = SelfHealingRAG(retriever, llm, verifier)

    queries = [
        "How does Raft elect a leader?",                       # should need no retrieval healing
        "why my cluster keep picking new boss all the time",   # casual phrasing -> likely rewrite
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        result = pipeline.answer(q)
        for e in result.log:
            print(f"  [{e.step}] ({e.kind}) {e.message}")
        print(f"A: {result.answer[:200]}")
        print(f"healing steps: {result.healing_steps}")

    print("\nSmoke test finished OK.")


if __name__ == "__main__":
    main()
