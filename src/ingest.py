"""Document ingestion: load text files, chunk them, embed, and build a FAISS index."""
import os
import json
import numpy as np
import faiss

from src import config


def load_documents(docs_dir: str = config.DOCS_DIR) -> list[dict]:
    """Load every .txt / .md file in docs_dir as a document."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith((".txt", ".md")):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, encoding="utf-8") as f:
            docs.append({"source": fname, "text": f.read()})
    return docs


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunking, snapping to sentence boundaries when possible."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Snap the cut to the last sentence end inside the window so chunks stay coherent.
        if end < len(text):
            snap = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if snap > start + size // 2:
                end = snap + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        # Don't start a chunk mid-word: advance to the next whitespace boundary.
        while start > 0 and start < len(text) and not text[start - 1].isspace():
            start += 1
    return chunks


def build_index(embedder, docs_dir: str = config.DOCS_DIR, index_dir: str = config.INDEX_DIR):
    """Chunk all docs, embed them, and persist a FAISS inner-product index (cosine on normalized vectors)."""
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"No .txt/.md documents found in {docs_dir}")

    chunks, metadata = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append(chunk)
            metadata.append({"source": doc["source"], "chunk_id": i, "text": chunk})

    embeddings = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "chunks.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    return index, metadata


def load_index(index_dir: str = config.INDEX_DIR):
    """Load a previously built index + metadata, or return None if absent."""
    faiss_path = os.path.join(index_dir, "chunks.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        return None, None
    index = faiss.read_index(faiss_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
