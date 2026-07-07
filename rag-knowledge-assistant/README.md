# RAG Knowledge Assistant

A small Retrieval-Augmented Generation (RAG) system that answers questions
over a set of internal documents. It indexes text into a local vector store
and answers questions by retrieving the most relevant chunks and handing them
to an LLM.

Built as a set of progressively more capable variants, all sharing the same
retrieval core (`rag_pipeline.py`):

| File | Generation backend | Retrieval style |
|---|---|---|
| `rag_pipeline.py` | Claude API (`claude-opus-4-8`) | Context is always stuffed into the prompt |
| `local_rag_pipeline.py` | Local Ollama model (`llama3.2:3b`) | Same as above, no API key needed |
| `agentic_rag.py` | Claude API | Retrieval exposed as a **tool** — the model decides if/when/how many times to search |
| `local_agentic_rag.py` | Local Ollama model | Same agentic loop, no API key needed |
| `api.py` | Local Ollama model | FastAPI wrapper exposing retrieval + Q&A over HTTP, with a persistent vector store |

## Architecture

```
documents (dict: source -> text)
        │
        ▼
  chunk_text()            word-based chunking, 500 words/chunk, 50 overlap
        │
        ▼
  Embedder                sentence-transformers (all-MiniLM-L6-v2), local, free
        │
        ▼
  VectorStore             FAISS (cosine/inner-product) + BM25Plus (keyword)
        │
        ├── search()          semantic only
        ├── keyword_search()  BM25 only
        └── hybrid_search()   reciprocal rank fusion of both
        │
        ▼
  build_prompt()          stuffs retrieved chunks into a context block
        │
        ▼
  answer_question()       Claude API  or  local Ollama model
```

The vector store can persist to disk (`VectorStore.save()` / `.load()`), so
documents only need to be embedded once — `api.py` uses this to survive
restarts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**For the Claude-backed variants** (`rag_pipeline.py`, `agentic_rag.py`):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**For the local variants** (`local_rag_pipeline.py`, `local_agentic_rag.py`, `api.py`):
```bash
brew services start ollama      # or: ollama serve
ollama pull llama3.2:3b
```

## Demo 1: standalone scripts (fastest way to show the concept)

Each script is self-contained — it seeds two sample docs (onboarding + deploy
policy) and asks a question end to end.

```bash
python rag_pipeline.py          # Claude API, single-shot retrieval
python local_rag_pipeline.py    # Ollama, single-shot retrieval, no API key
python agentic_rag.py           # Claude API, model decides when to search
python local_agentic_rag.py     # Ollama, agentic loop, no API key
```

Expected output shape:
```
Q: Can I deploy to production on a Friday?
A: Only with sign-off from an engineering lead — Friday deploys require...
```

## Demo 2: the API (persistent, multi-document, hybrid search)

Start the server:
```bash
source .venv/bin/activate
uvicorn api:app --reload
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","total_chunks":0}
```

Index documents (one-time, persists to `vector_db/`):
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": {
      "onboarding.md": "New engineers get a laptop, a GitHub invite, and access to the internal wiki on their first day. Slack channels #eng-general and #eng-help are the main places to ask questions.",
      "deploy-policy.md": "Deploys to production happen Monday through Thursday, 10am-4pm. Friday deploys require sign-off from an engineering lead."
    }
  }'
```

Search (semantic / keyword / hybrid):
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "friday deploy rules", "k": 3, "mode": "hybrid"}'
```

Ask a question (retrieval + generation):
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I deploy to production on a Friday?"}'
```

### API reference

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/health` | — | liveness + chunk count |
| POST | `/documents` | `{"documents": {source: text}}` | indexes and persists to `vector_db/` |
| POST | `/search` | `{"query", "k", "mode"}` | `mode`: `semantic` \| `keyword` \| `hybrid` |
| POST | `/ask` | `{"question"}` | hybrid retrieval → local Ollama answer |

## Talking points for the demo

- **Hybrid search**: semantic (embeddings) catches paraphrases; BM25 keyword
  search catches exact terms/acronyms embeddings miss. Combined via
  reciprocal rank fusion in `VectorStore.hybrid_search()`.
- **Swappable generation backend**: same retrieval core works with Claude
  (cloud, best quality) or Ollama (local, free, offline) — useful for
  demoing without burning API credits.
- **Agentic vs. single-shot**: `agentic_rag.py` shows the model deciding
  *whether* and *how many times* to call search as a tool, versus always
  stuffing context into the prompt — better for multi-part questions.
- **Persistence**: the API's vector store survives restarts, so a live demo
  doesn't need to re-embed documents every time.

## Known limitations (be upfront about these)

- No chunk deduplication — re-indexing the same document twice duplicates it.
- No delete/update endpoint — the store is append-only.
- `local_agentic_rag.py` requires an Ollama model that supports tool calling.
- Single-process, in-memory FAISS index — not built for concurrent writers
  or large-scale production traffic.
