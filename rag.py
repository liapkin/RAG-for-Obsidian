#!/usr/bin/env python3
"""RAG over an Obsidian vault: local embeddings (fastembed) + DeepSeek for answers.

Usage:
    python rag.py index         # (re)build the index from the vault
    python rag.py "question"    # ask a question
    python rag.py selftest      # sanity-check chunking + retrieval
"""
import json
import os
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


def ask_deepseek(prompt: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        sys.exit(f"DeepSeek API error {e.code}: {e.read().decode(errors='replace')}")


def query(question: str):
    if not INDEX_NPZ.exists():
        sys.exit("no index — run: python rag.py index")
    vecs = np.load(INDEX_NPZ)["vecs"]
    chunks = json.loads(INDEX_JSON.read_text())
    qvec = embed([question])[0]
    top = np.argsort(vecs @ qvec)[::-1][:TOP_K]
    context = "\n\n---\n\n".join(
        f"[{chunks[i]['file']}#{chunks[i]['heading']}]\n{chunks[i]['text']}" for i in top
    )
    prompt = (
        "Answer the question using only the notes below. Cite the [file#heading] you used. "
        f"If the notes don't cover it, say so.\n\nNotes:\n{context}\n\nQuestion: {question}"
    )
    print(ask_deepseek(prompt))
    print("\nSources:")
    for i in top:
        print(f"  {chunks[i]['file']}#{chunks[i]['heading']}")


def selftest():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("intro text\n\n## Alpha\nalpha body\n\n## Beta\n" + "x" * 2000 + "\n\nbeta tail\n")
        p = Path(f.name)
    cs = chunk_file(p)
    p.unlink()
    assert [c["heading"] for c in cs] == [p.stem, "Alpha", "Beta", "Beta"], cs
    assert "alpha body" in cs[1]["text"]

    vecs = embed(["the capital of France is Paris", "how to bake sourdough bread"])
    q = embed(["french capital city"])[0]
    assert int(np.argmax(vecs @ q)) == 0
    print("selftest ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "index":
        build_index()
    elif args[0] == "selftest":
        selftest()
    else:
        query(" ".join(args))
