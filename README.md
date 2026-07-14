# RAG for Obsidian

Minimal RAG over an Obsidian vault: local embeddings ([fastembed](https://github.com/qdrant/fastembed), `bge-small-en-v1.5`) + BM25 hybrid retrieval + streaming DeepSeek answers. No vector DB, no SDKs, no frameworks.

## Quick start (Docker)

```bash
cp .env.example .env         # put your DEEPSEEK_API_KEY in it
VAULT_DIR=/path/to/vault docker compose up    # defaults to ../obsidian
# open http://localhost:8000
```

Indexes the vault automatically on first start.

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env
ln -s /path/to/your/vault vault
python -m rag serve          # web UI at localhost:8000
```

## CLI

```bash
python -m rag index          # (re)build the index
python -m rag "how do I restore a database?"
python -m rag chat           # REPL with conversation history
python -m rag selftest       # sanity-check chunking + retrieval
```

## Structure

```
rag/
  config.py      # paths, model, tunables, .env loader
  chunking.py    # markdown -> chunks (split on ## headings)
  indexing.py    # embed + save/load the index (npz + json)
  retrieval.py   # BM25 + dense cosine, fused with reciprocal rank fusion
  llm.py         # DeepSeek streaming + follow-up query rewriting
  query.py       # RAG pipeline: retrieve -> prompt -> stream, CLI chat
  server.py      # stdlib http.server: serves index.html + streaming /ask
  index.html     # chat UI, renders markdown (marked.js), localStorage history
```

## How it works

1. **Chunk**: each `.md` split on `## ` headings; oversized sections split on blank lines (~1500 chars).
2. **Embed**: chunks embedded locally, normalized, stored in `index.npz` + `index.json`.
3. **Retrieve**: hybrid — dense (cosine = one numpy dot product) + sparse (BM25, ~20 lines) rankings fused with reciprocal rank fusion. Vectors catch paraphrases, BM25 catches exact terms (error codes, CLI flags).
4. **Rewrite follow-ups**: in conversations, "how do I drop it?" is condensed into a standalone query with one cheap LLM call before retrieval.
5. **Generate**: top 5 chunks + question streamed from DeepSeek with forced `[file#heading]` citations; answers "not covered" when the notes don't have it.

No vector DB by design — a few hundred chunks don't need one. Past ~10k chunks: FAISS/Chroma. Better retrieval: cross-encoder reranking.
