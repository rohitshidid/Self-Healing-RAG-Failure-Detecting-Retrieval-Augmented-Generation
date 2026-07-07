"""NLI-based verification: chunk relevance and answer groundedness."""
import re

from src import config


class Verifier:
    """Two cross-encoders:

    - relevance_model (ms-marco): scores (query, passage) relevance as a raw logit.
    - nli_model (nli-deberta-v3-base): logits over [contradiction, entailment, neutral],
      used for claim groundedness.
    """

    ENTAILMENT_IDX = 1

    def __init__(self, nli_model, relevance_model):
        self.model = nli_model
        self.relevance_model = relevance_model

    def _entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np
        logits = self.model.predict(pairs, show_progress_bar=False)
        logits = np.asarray(logits)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, self.ENTAILMENT_IDX].tolist()

    # --- Chunk relevance ---------------------------------------------------
    def chunk_relevance(self, question: str, chunks: list[dict]) -> list[float]:
        """Score each chunk's relevance to the question (ms-marco raw logits)."""
        if not chunks:
            return []
        scores = self.relevance_model.predict(
            [(question, c["text"]) for c in chunks], show_progress_bar=False)
        return [float(s) for s in scores]

    # --- Groundedness ------------------------------------------------------
    @staticmethod
    def split_claims(answer: str) -> list[str]:
        """Split an answer into sentence-level claims."""
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    @staticmethod
    def _premises(chunks: list[dict]) -> list[str]:
        """Chunk texts plus 2-sentence sliding windows within each chunk.

        DeBERTa NLI under-scores evidence buried mid-way in a long premise, so short
        windows recover entailment the full chunk misses; the full chunk still catches
        claims whose evidence spans many sentences.
        """
        premises = []
        for c in chunks:
            premises.append(c["text"])
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c["text"]) if len(s.strip()) > 20]
            for i in range(len(sents)):
                premises.append(" ".join(sents[i:i + 2]))
        return premises

    def check_groundedness(self, answer: str, chunks: list[dict]) -> dict:
        """Verify each claim in the answer is entailed by the retrieved context.

        Returns {"claims": [{"claim", "score", "grounded"}], "all_grounded": bool}.
        """
        claims = self.split_claims(answer)
        if not claims or not chunks:
            return {"claims": [], "all_grounded": bool(claims == [])}

        premises = self._premises(chunks)
        results = []
        for claim in claims:
            pairs = [(p, claim) for p in premises]
            probs = self._entailment_probs(pairs)
            best = max(probs)
            results.append({
                "claim": claim,
                "score": best,
                "grounded": best >= config.GROUNDEDNESS_THRESHOLD,
            })
        return {
            "claims": results,
            "all_grounded": all(r["grounded"] for r in results),
        }
