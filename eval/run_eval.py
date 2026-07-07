"""Compare self-healing pipeline vs. baseline RAG on the eval set.

Baseline = retrieve -> generate -> return (no detection, no healing).
Metrics: answered rate, groundedness rate, healing activations.

Usage:  GROQ_API_KEY=... python -m eval.run_eval
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.generation import LLMClient, is_non_answer
from src.ingest import build_index, load_index
from src.pipeline import SelfHealingRAG
from src.retrieval import Retriever
from src.verification import Verifier


def baseline_answer(retriever, llm, verifier, question):
    """Plain RAG: no failure detection, no healing."""
    chunks = retriever.retrieve(question)
    answer = llm.generate_answer(question, chunks)
    grounded = None
    if not is_non_answer(answer):
        grounded = verifier.check_groundedness(answer, chunks)["all_grounded"]
    return answer, grounded


def main():
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY first.")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("Loading models...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    nli = CrossEncoder(config.NLI_MODEL)
    relevance = CrossEncoder(config.RELEVANCE_MODEL)

    index, metadata = load_index()
    if index is None:
        index, metadata = build_index(embedder)

    retriever = Retriever(embedder, index, metadata)
    llm = LLMClient()
    verifier = Verifier(nli, relevance)
    pipeline = SelfHealingRAG(retriever, llm, verifier)

    with open(os.path.join(os.path.dirname(__file__), "eval_set.json")) as f:
        eval_set = json.load(f)

    stats = {
        "baseline": {"answered": 0, "grounded": 0, "gradable": 0},
        "healing": {"answered": 0, "grounded": 0, "gradable": 0, "heals": 0},
    }
    rows = []

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        print(f"\n[{i}/{len(eval_set)}] {q}")

        b_answer, b_grounded = baseline_answer(retriever, llm, verifier, q)
        b_answered = not is_non_answer(b_answer)

        result = pipeline.answer(q)
        h_answered = not is_non_answer(result.answer)
        h_grounded = result.groundedness["all_grounded"] if result.groundedness else None

        if item["expect_answer"]:
            stats["baseline"]["answered"] += b_answered
            stats["healing"]["answered"] += h_answered
            if b_grounded is not None:
                stats["baseline"]["gradable"] += 1
                stats["baseline"]["grounded"] += b_grounded
            if h_grounded is not None:
                stats["healing"]["gradable"] += 1
                stats["healing"]["grounded"] += h_grounded
        stats["healing"]["heals"] += result.healing_steps

        rows.append({
            "question": q,
            "expect_answer": item["expect_answer"],
            "baseline": {"answered": b_answered, "grounded": b_grounded, "answer": b_answer},
            "healing": {"answered": h_answered, "grounded": h_grounded,
                        "healing_steps": result.healing_steps, "answer": result.answer},
        })
        print(f"  baseline: answered={b_answered} grounded={b_grounded}")
        print(f"  healing : answered={h_answered} grounded={h_grounded} "
              f"steps={result.healing_steps}")

    expected = sum(1 for x in eval_set if x["expect_answer"])
    print("\n" + "=" * 60)
    print(f"Questions expecting an answer: {expected}")
    print(f"Baseline    — answered: {stats['baseline']['answered']}/{expected}, "
          f"fully grounded: {stats['baseline']['grounded']}/{stats['baseline']['gradable']}")
    print(f"Self-healing — answered: {stats['healing']['answered']}/{expected}, "
          f"fully grounded: {stats['healing']['grounded']}/{stats['healing']['gradable']}, "
          f"total healing steps: {stats['healing']['heals']}")

    out = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out, "w") as f:
        json.dump({"stats": stats, "rows": rows}, f, indent=2)
    print(f"\nDetailed results written to {out}")


if __name__ == "__main__":
    main()
