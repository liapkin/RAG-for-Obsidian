# RAG for Obsidian

Minimal RAG over an Obsidian vault: local embeddings ([fastembed](https://github.com/qdrant/fastembed), `bge-small-en-v1.5`) + DeepSeek for answer generation. One file, no vector DB, no SDKs.

## Setup

```bash
pip install fastembed numpy
cp .env.example .env        # put your DEEPSEEK_API_KEY in it
ln -s /path/to/your/vault vault
```

## Usage

```bash
python rag.py index          # chunk + embed the vault (markdown, split on ## headings)
python rag.py "how do I restore a database?"
python rag.py selftest       # sanity-check chunking + retrieval
```

## How it works

1. **Chunk**: each `.md` split on `## ` headings; oversized sections split on blank lines (~1500 chars).
2. **Embed**: chunks embedded locally, normalized, stored in `index.npz` + `index.json`.
3. **Retrieve**: question embedded, cosine similarity = one numpy dot product, top 5 chunks.
4. **Generate**: chunks + question sent to DeepSeek with forced `[file#heading]` citations; answers "not covered" when the notes don't have it.

No vector DB by design — a few hundred chunks don't need one. Past ~10k chunks: FAISS/Chroma. Better retrieval: hybrid BM25 + rerank.
