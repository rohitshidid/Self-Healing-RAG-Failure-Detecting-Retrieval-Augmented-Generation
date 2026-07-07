"""Vector retrieval over the FAISS index."""
import numpy as np

from src import config


class Retriever:
    def __init__(self, embedder, index, metadata):
        self.embedder = embedder
        self.index = index
        self.metadata = metadata

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """Return top_k chunks with cosine similarity scores, highest first."""
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        top_k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "score": float(score),
            })
        return results

    def retrieve_by_embedding(self, embedding: np.ndarray, top_k: int = config.TOP_K) -> list[dict]:
        """Retrieve using a precomputed (normalized) embedding — used by HyDE."""
        q = np.asarray([embedding], dtype="float32")
        top_k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(q, top_k)
        return [
            {
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"],
                "score": float(score),
            }
            for score, idx in zip(scores[0], ids[0])
            if idx != -1
        ]


def avg_similarity(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    return sum(c["score"] for c in chunks) / len(chunks)
