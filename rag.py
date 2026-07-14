#!/usr/bin/env python3
"""RAG over an Obsidian vault: local embeddings (fastembed) + DeepSeek for answers.

Retrieval is hybrid: dense (embeddings, cosine) + sparse (BM25) fused with
reciprocal rank fusion. Answers stream from DeepSeek.

Usage:
    python rag.py index         # (re)build the index from the vault
    python rag.py "question"    # ask one question
    python rag.py chat          # interactive REPL with conversation history
    python rag.py selftest      # sanity-check chunking + retrieval
"""
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ponytail: 5-line .env loader, swap for python-dotenv if quoting/expansion ever needed
if (HERE / ".env").exists():
    for _line in (HERE / ".env").read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

VAULT = Path(os.environ.get("VAULT", HERE / "vault"))
INDEX_NPZ = HERE / "index.npz"
INDEX_JSON = HERE / "index.json"
MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
MAX_CHUNK = 1500


def chunk_file(path: Path) -> list[dict]:
    """Split a markdown file on ## headings; split oversized chunks on blank lines."""
    text = path.read_text(errors="replace")
    chunks = []
    heading = path.stem
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        # split oversized sections on blank lines
        piece = ""
        for para in body.split("\n\n"):
            if piece and len(piece) + len(para) > MAX_CHUNK:
                chunks.append({"text": piece.strip(), "file": path.name, "heading": heading})
                piece = para
            else:
                piece = f"{piece}\n\n{para}" if piece else para
        if piece.strip():
            chunks.append({"text": piece.strip(), "file": path.name, "heading": heading})

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            buf = []
            heading = line[3:].strip()
        buf.append(line)
    flush()
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding  # lazy: slow import, not needed for selftest of chunker

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rank(question: str, chunks: list[dict], k1=1.5, b=0.75) -> np.ndarray:
    """Chunk indices ranked by BM25, best first.

    ponytail: recomputed per query, fine at hundreds of chunks — precompute at index time if slow.
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


def ask_deepseek(messages: list[dict]) -> str:
    """Stream the answer to stdout, return the full text."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": "deepseek-chat", "messages": messages, "stream": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    parts = []
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                parts.append(delta)
                print(delta, end="", flush=True)
    except urllib.error.HTTPError as e:
        sys.exit(f"DeepSeek API error {e.code}: {e.read().decode(errors='replace')}")
    print()
    return "".join(parts)


def query(question: str, history: list[dict] | None = None) -> list[dict]:
    if not INDEX_NPZ.exists():
        sys.exit("no index — run: python rag.py index")
    vecs = np.load(INDEX_NPZ)["vecs"]
    chunks = json.loads(INDEX_JSON.read_text())
    top = retrieve(question, vecs, chunks)
    context = "\n\n---\n\n".join(
        f"[{chunks[i]['file']}#{chunks[i]['heading']}]\n{chunks[i]['text']}" for i in top
    )
    system = {
        "role": "system",
        "content": "Answer from the retrieved notes and the conversation history — nothing else. "
        "Your own previous answers are a fully valid source: for follow-ups, ignore the freshly "
        "retrieved notes if the conversation already holds the answer. Cite the [file#heading] "
        "you used. If nothing covers it, say so.",
    }
    prompt = f"Notes:\n{context}\n\nQuestion: {question}"
    messages = [system] + (history or []) + [{"role": "user", "content": prompt}]
    answer = ask_deepseek(messages)
    print("\nSources:")
    for i in top:
        print(f"  {chunks[i]['file']}#{chunks[i]['heading']}")
    # keep history lean: store the bare question, not the injected notes
    return (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def chat():
    # ponytail: retrieval sees only the bare follow-up ("how do I drop it?"), not the
    # conversation — upgrade path: rewrite the query with history via one extra LLM call.
    history: list[dict] = []
    print("obsidian rag — ask away (ctrl-d to quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if question:
            history = query(question, history)


def selftest():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("intro text\n\n## Alpha\nalpha body\n\n## Beta\n" + "x" * 2000 + "\n\nbeta tail\n")
        p = Path(f.name)
    cs = chunk_file(p)
    p.unlink()
    assert [c["heading"] for c in cs] == [p.stem, "Alpha", "Beta", "Beta"], cs
    assert "alpha body" in cs[1]["text"]

    toy = [{"text": "the capital of France is Paris"}, {"text": "how to bake sourdough bread"}]
    assert int(bm25_rank("sourdough bread recipe", toy)[0]) == 1

    vecs = embed([c["text"] for c in toy])
    q = embed(["french capital city"])[0]
    assert int(np.argmax(vecs @ q)) == 0
    assert retrieve("bake sourdough", vecs, toy)[0] == 1  # hybrid agrees when both signals do
    print("selftest ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "index":
        build_index()
    elif args[0] == "chat":
        chat()
    elif args[0] == "selftest":
        selftest()
    else:
        query(" ".join(args))
