"""Healing strategies, one per failure type.

Each strategy takes the current pipeline state and returns new retrieved chunks
(and optionally a note describing what it did). The pipeline decides whether the
heal succeeded.
"""
import numpy as np

from src import config


def heal_low_similarity(llm, retriever, question: str) -> tuple[list[dict], str, str]:
    """Query rewriting: rephrase the query with the LLM and retry retrieval."""
    rewritten = llm.rewrite_query(question)
    chunks = retriever.retrieve(rewritten)
    return chunks, f'rewrote query to: "{rewritten}"', rewritten


def heal_wrong_chunks(llm, retriever, question: str) -> tuple[list[dict], str]:
    """HyDE: generate a hypothetical answer, embed it, retrieve with that embedding."""
    hypothetical = llm.hyde_document(question)
    emb = retriever.embedder.encode([hypothetical], normalize_embeddings=True)[0]
    chunks = retriever.retrieve_by_embedding(np.asarray(emb, dtype="float32"))
    preview = hypothetical[:120] + ("..." if len(hypothetical) > 120 else "")
    return chunks, f'HyDE hypothetical doc: "{preview}"'


def heal_ungrounded(llm, retriever, question: str, claim: str,
                    current_chunks: list[dict]) -> tuple[list[dict], str]:
    """Re-retrieval: search for the ungrounded claim directly, merge with current context."""
    refined = llm.refine_query_for_claim(question, claim)
    extra = retriever.retrieve(refined)
    seen = {c["text"] for c in current_chunks}
    merged = current_chunks + [c for c in extra if c["text"] not in seen]
    return merged[:config.EXPANDED_TOP_K], f'searched for claim evidence with: "{refined}"'


def heal_empty_answer(retriever, question: str) -> tuple[list[dict], str]:
    """Retrieval expansion: widen the search (more chunks) and retry generation."""
    chunks = retriever.retrieve(question, top_k=config.EXPANDED_TOP_K)
    return chunks, f"expanded retrieval to top-{config.EXPANDED_TOP_K} chunks"
