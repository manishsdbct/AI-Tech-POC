"""
Basic RAG (Retrieval-Augmented Generation) pipeline.

Pattern: embed docs -> embed query -> retrieve top-k by cosine similarity
-> stuff retrieved chunks into the prompt -> generate with Claude.

Embeddings run locally (sentence-transformers) since the Claude API has no
embeddings endpoint. Swap `Embedder` for Voyage AI (Anthropic's recommended
embeddings partner) or any other provider without touching the rest of the
pipeline.
"""

import json
import os
import faiss
import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

MODEL_ID = "claude-opus-4-8"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class VectorStore:
    """Semantic index backed by FAISS. Embeddings persist to disk via save()/load()
    so keys don't need to be re-embedded on every run."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Plus | None = None

    def add_documents(self, documents: dict[str, str]) -> None:
        new_chunks, new_sources = [], []
        for source, text in documents.items():
            for chunk in chunk_text(text):
                new_chunks.append(chunk)
                new_sources.append(source)

        vectors = np.asarray(self.embedder.embed(new_chunks), dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (vectors are normalized)
        self.index.add(vectors)

        self.chunks.extend(new_chunks)
        self.sources.extend(new_sources)
        self.bm25 = BM25Plus([c.lower().split() for c in self.chunks])

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """Semantic search: cosine similarity over embeddings via FAISS."""
        query_vec = np.asarray(self.embedder.embed([query]), dtype="float32")
        scores, idx = self.index.search(query_vec, min(k, len(self.chunks)))
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(s)}
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]

    def save(self, path: str) -> None:
        """Persist the FAISS index + chunk/source metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"chunks": self.chunks, "sources": self.sources}, f)

    @classmethod
    def load(cls, path: str, embedder: Embedder) -> "VectorStore":
        """Load a previously saved FAISS index + metadata from disk."""
        store = cls(embedder)
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json")) as f:
            meta = json.load(f)
        store.chunks = meta["chunks"]
        store.sources = meta["sources"]
        store.bm25 = BM25Plus([c.lower().split() for c in store.chunks])
        return store

    def keyword_search(self, query: str, k: int = TOP_K) -> list[dict]:
        """Lexical search via BM25 - good at matching exact terms embeddings can miss."""
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(-scores)[:k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def hybrid_search(self, query: str, k: int = TOP_K, rrf_k: int = 60) -> list[dict]:
        """Combine semantic + keyword search via reciprocal rank fusion."""
        semantic = self.search(query, k=len(self.chunks))
        keyword = self.keyword_search(query, k=len(self.chunks))

        rrf_scores: dict[int, float] = {}
        for rank, r in enumerate(semantic):
            idx = self.chunks.index(r["text"])
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)
        for rank, r in enumerate(keyword):
            idx = self.chunks.index(r["text"])
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

        top_idx = sorted(rrf_scores, key=lambda i: -rrf_scores[i])[:k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": rrf_scores[i]}
            for i in top_idx
        ]


def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    return (
        f"Use the following retrieved context to answer the question. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {query}"
    )


def answer_question(client: Anthropic, store: VectorStore, query: str) -> str:
    retrieved = store.hybrid_search(query)
    prompt = build_prompt(query, retrieved)

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system="You are a helpful assistant that answers questions using only the provided context.",
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    documents = {
        "onboarding.md": (
            "New engineers get a laptop, a GitHub invite, and access to the internal wiki "
            "on their first day. Slack channels #eng-general and #eng-help are the main "
            "places to ask questions. IT support is reachable at it-help@example.com."
        ),
        "deploy-policy.md": (
            "Deploys to production happen Monday through Thursday, 10am-4pm, to ensure "
            "someone is available to respond if something breaks. Friday deploys require "
            "sign-off from an engineering lead. All deploys go through the CI pipeline; "
            "manual deploys to prod are disabled."
        ),
    }

    embedder = Embedder()
    store = VectorStore(embedder)
    store.add_documents(documents)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    query = "Can I deploy to production on a Friday?"
    print(f"Q: {query}")
    print(f"A: {answer_question(client, store, query)}")
