"""
Local-LLM variant of agentic_rag.py: same tool-use ("agentic RAG") loop,
but generation runs through a local Ollama model instead of the
Anthropic API. Retrieval is exposed as a tool and the model decides
whether and how many times to call it before answering.

Requires an Ollama server running locally (`ollama serve` or
`brew services start ollama`) with LOCAL_MODEL_ID pulled, and a model
that supports tool calling.
"""

import json
import ollama
from rag_pipeline import Embedder, VectorStore

LOCAL_MODEL_ID = "llama3.2:3b"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base for information relevant to a query. "
            "Call this whenever you need facts from company docs to answer the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
}


def run_agentic_rag(store: VectorStore, user_question: str, model: str = LOCAL_MODEL_ID) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer questions using the search_knowledge_base tool to look up facts before responding.",
        },
        {"role": "user", "content": user_question},
    ]

    while True:
        response = ollama.chat(model=model, messages=messages, tools=[SEARCH_TOOL])
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            results = store.hybrid_search(call.function.arguments["query"])
            messages.append({
                "role": "tool",
                "content": json.dumps(results),
            })


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

    store = VectorStore(Embedder())
    store.add_documents(documents)

    question = "What's the deploy policy, and who do I contact for IT help?"
    print(f"Q: {question}")
    print(f"A: {run_agentic_rag(store, question)}")
