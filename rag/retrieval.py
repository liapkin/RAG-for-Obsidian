import math
import re

import numpy as np

from .config import TOP_K
from .indexing import embed


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rank(question: str, chunks: list[dict], k1=1.5, b=0.75) -> np.ndarray:
    """Chunk indices ranked by BM25, best first.

    recomputed per query, fine at hundreds of chunks — precompute at index time if slow.
    """
    docs = [tokenize(c["text"]) for c in chunks]
    avgdl = sum(map(len, docs)) / len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    scores = np.zeros(n)
    for t in tokenize(question):
        if t not in df:
            continue
        idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for i, d in enumerate(docs):
            tf = d.count(t)
            scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return np.argsort(scores)[::-1]


def retrieve(question: str, vecs: np.ndarray, chunks: list[dict]) -> list[int]:
    """Hybrid retrieval: dense + BM25 rankings fused with reciprocal rank fusion."""
    dense = np.argsort(vecs @ embed([question])[0])[::-1]
    sparse = bm25_rank(question, chunks)
    rrf: dict[int, float] = {}
    for ranking in (dense, sparse):
        for rank, i in enumerate(ranking):
            rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
    return sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]
