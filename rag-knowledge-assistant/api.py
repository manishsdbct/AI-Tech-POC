"""
FastAPI wrapper around the local (Ollama-backed) RAG pipeline.

Endpoints:
  GET  /health     - liveness check
  POST /documents  - add keys/documents to the persistent vector store
  POST /search     - run semantic / keyword / hybrid retrieval
  POST /ask        - retrieve + generate an answer via the local Ollama model

The vector store persists to disk (VECTOR_DB_PATH) across restarts via
VectorStore.save()/load(), so documents only need to be indexed once.

Run with: uvicorn api:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_rag_pipeline import answer_question
from rag_pipeline import Embedder, VectorStore

VECTOR_DB_PATH = "vector_db"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder()
    if os.path.exists(VECTOR_DB_PATH):
        state["store"] = VectorStore.load(VECTOR_DB_PATH, embedder)
    else:
        state["store"] = VectorStore(embedder)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


class AddDocumentsRequest(BaseModel):
    documents: dict[str, str]  # key/source -> text


class SearchRequest(BaseModel):
    query: str
    k: int = 3
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"


class AskRequest(BaseModel):
    question: str


def get_store() -> VectorStore:
    store: VectorStore = state["store"]
    if not store.chunks:
        raise HTTPException(400, "No documents indexed yet. POST /documents first.")
    return store


@app.get("/health")
def health():
    return {"status": "ok", "total_chunks": len(state["store"].chunks)}


@app.post("/documents")
def add_documents(req: AddDocumentsRequest):
    store: VectorStore = state["store"]
    store.add_documents(req.documents)
    store.save(VECTOR_DB_PATH)
    return {"added_keys": len(req.documents), "total_chunks": len(store.chunks)}


@app.post("/search")
def search(req: SearchRequest):
    store = get_store()
    if req.mode == "semantic":
        results = store.search(req.query, k=req.k)
    elif req.mode == "keyword":
        results = store.keyword_search(req.query, k=req.k)
    else:
        results = store.hybrid_search(req.query, k=req.k)
    return {"results": results}


@app.post("/ask")
def ask(req: AskRequest):
    store = get_store()
    return {"answer": answer_question(store, req.question)}
