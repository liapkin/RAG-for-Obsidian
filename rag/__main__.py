"""RAG over an Obsidian vault: local embeddings (fastembed) + DeepSeek for answers.

Retrieval is hybrid: dense (embeddings, cosine) + sparse (BM25) fused with
reciprocal rank fusion. Answers stream from DeepSeek.

Usage:
    python -m rag index         # (re)build the index from the vault
    python -m rag "question"    # ask one question
    python -m rag chat          # interactive REPL with conversation history
    python -m rag serve [port]  # web UI with rendered markdown (default :8000)
    python -m rag selftest      # sanity-check chunking + retrieval
"""
import sys


def selftest():
    import tempfile
    from pathlib import Path

    import numpy as np

    from .chunking import chunk_file
    from .indexing import embed
    from .retrieval import bm25_rank, retrieve

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


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "index":
        from .indexing import build_index

        build_index()
    elif args[0] == "chat":
        from .query import chat

        chat()
    elif args[0] == "serve":
        from .server import serve

        serve(int(args[1]) if len(args) > 1 else 8000)
    elif args[0] == "selftest":
        selftest()
    else:
        from .query import query

        query(" ".join(args))


main()
