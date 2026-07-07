"""
Local-LLM variant of rag_pipeline.py: same retrieval logic (local
sentence-transformers embeddings + cosine similarity search), but
generation runs through a local Ollama model instead of the Anthropic
API. Useful for running the example without an ANTHROPIC_API_KEY.

Requires an Ollama server running locally (`ollama serve` or
`brew services start ollama`) with LOCAL_MODEL_ID pulled.
"""

import ollama
from rag_pipeline import Embedder, VectorStore, build_prompt

LOCAL_MODEL_ID = "llama3.2:3b"


def answer_question(store: VectorStore, query: str, model: str = LOCAL_MODEL_ID) -> str:
    retrieved = store.hybrid_search(query)
    prompt = build_prompt(query, retrieved)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using only the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content


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

    query = "Can I deploy to production on a Friday?"
    print(f"Q: {query}")
    print(f"A: {answer_question(store, query)}")
