# AI-Tech-POC

Proof-of-concept projects.

## Projects

- [rag-knowledge-assistant](rag-knowledge-assistant/) — Retrieval-Augmented Generation Q&A system over internal documents, with Claude and local-Ollama variants and a FastAPI wrapper.
- [llm-gateway](llm-gateway/) — Central FastAPI gateway for internal services to call OpenAI/Anthropic through, with auth, per-service rate limits, response caching, cost-aware model routing with fallback, and Postgres-backed budgets/usage tracking.
