"""LLM calls via Groq: answer generation, query rewriting, HyDE document generation."""
import os
import time

from groq import Groq, RateLimitError

from src import config

_ANSWER_SYSTEM = (
    "You are a precise assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Answer in 2-4 short declarative sentences. State facts directly, staying close to "
    "the context's wording. Do not hedge (no 'may', 'might', 'could'), do not comment "
    "on the context itself, and do not use outside knowledge."
)

_REWRITE_SYSTEM = (
    "You rewrite search queries to better match a document corpus. "
    "Given a question, produce ONE rephrased version using more specific, formal, "
    "domain-relevant vocabulary. Output only the rewritten query, nothing else."
)

_HYDE_SYSTEM = (
    "Write a short, factual paragraph (3-4 sentences) that would plausibly answer "
    "the user's question, as if it were an excerpt from a technical document. "
    "Output only the paragraph."
)


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = config.GROQ_MODEL):
        self.model = model
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 5)  # 5s, 10s, 20s
        raise RuntimeError("unreachable")

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self._chat(_ANSWER_SYSTEM, prompt)

    def rewrite_query(self, question: str) -> str:
        return self._chat(_REWRITE_SYSTEM, question, temperature=0.4)

    def hyde_document(self, question: str) -> str:
        return self._chat(_HYDE_SYSTEM, question, temperature=0.5)

    def refine_query_for_claim(self, question: str, claim: str) -> str:
        prompt = (
            f"Original question: {question}\n"
            f"Unverified claim from a draft answer: {claim}\n"
            "Write ONE short search query that would find evidence for or against this claim."
        )
        return self._chat(_REWRITE_SYSTEM, prompt, temperature=0.3)


def is_non_answer(answer: str) -> bool:
    """Detect empty / 'I don't know' style answers."""
    a = answer.strip().lower().rstrip(".!")
    if not a:
        return True
    refusals = ("i don't know", "i do not know", "i dont know", "cannot answer",
                "not enough information", "no information")
    return any(a.startswith(r) or r in a[:80] for r in refusals)
