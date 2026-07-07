"""Deterministic fakes for unit-testing the pipeline without models or API keys."""
import hashlib

import numpy as np


class FakeEmbedder:
    """Maps known strings to fixed unit vectors so similarity is controllable."""

    def __init__(self, vector_map: dict[str, np.ndarray], dim: int = 64, default_scale: float = 0.1):
        self.vector_map = vector_map
        self.dim = dim
        self.default_scale = default_scale

    def _vec(self, text: str) -> np.ndarray:
        for key, vec in self.vector_map.items():
            if key in text.lower():
                v = np.zeros(self.dim)
                v[:len(vec)] = vec
                return v / np.linalg.norm(v)
        # Unknown text: deterministic pseudo-random unit vector, far from corpus
        # (md5, not hash(): hash() is salted per process and would make tests flaky).
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim)
        return v / np.linalg.norm(v)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([self._vec(t) for t in texts], dtype="float32")


class FakeLLM:
    """Scripted LLM: configurable responses per method."""

    def __init__(self, answers=None, rewrite="rewritten query about topic",
                 hyde="hypothetical document about topic", refined="refined claim query"):
        # answers: list popped in order across generate_answer calls
        self.answers = list(answers or ["A grounded answer about the topic."])
        self.rewrite_result = rewrite
        self.hyde_result = hyde
        self.refined_result = refined
        self.calls = []

    def generate_answer(self, question, chunks):
        self.calls.append(("generate", question, len(chunks)))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def rewrite_query(self, question):
        self.calls.append(("rewrite", question))
        return self.rewrite_result

    def hyde_document(self, question):
        self.calls.append(("hyde", question))
        return self.hyde_result

    def refine_query_for_claim(self, question, claim):
        self.calls.append(("refine", claim))
        return self.refined_result


class FakeVerifier:
    """Scripted verifier: fixed relevance scores and groundedness verdicts."""

    def __init__(self, relevance=5.0, groundedness_sequence=None):
        self.relevance = relevance
        # groundedness_sequence: list of bools consumed per check_groundedness call
        self.groundedness_sequence = list(groundedness_sequence or [True])

    def chunk_relevance(self, question, chunks):
        return [self.relevance] * len(chunks)

    def check_groundedness(self, answer, chunks):
        grounded = (self.groundedness_sequence.pop(0)
                    if len(self.groundedness_sequence) > 1
                    else self.groundedness_sequence[0])
        claims = [{"claim": s, "score": 0.9 if grounded else 0.1, "grounded": grounded}
                  for s in answer.split(". ") if len(s) > 15]
        return {"claims": claims, "all_grounded": all(c["grounded"] for c in claims)}
