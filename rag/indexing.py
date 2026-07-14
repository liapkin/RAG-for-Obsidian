import json
import sys

import numpy as np

from .chunking import chunk_file
from .config import INDEX_JSON, INDEX_NPZ, MODEL, VAULT


def embed(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding  # lazy: slow import

    model = TextEmbedding(MODEL)
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # normalized -> cosine == dot


def build_index():
    if not VAULT.is_dir():
        sys.exit(f"vault not found: {VAULT}")
    chunks = [c for f in sorted(VAULT.rglob("*.md")) for c in chunk_file(f)]
    if not chunks:
        sys.exit(f"no markdown chunks found in {VAULT}")
    vecs = embed([c["text"] for c in chunks])
    np.savez(INDEX_NPZ, vecs=vecs)
    INDEX_JSON.write_text(json.dumps(chunks))
    print(f"indexed {len(chunks)} chunks from {len(set(c['file'] for c in chunks))} files")


def load_index() -> tuple[np.ndarray, list[dict]]:
    if not INDEX_NPZ.exists():
        sys.exit("no index — run: python -m rag index")
    return np.load(INDEX_NPZ)["vecs"], json.loads(INDEX_JSON.read_text())
