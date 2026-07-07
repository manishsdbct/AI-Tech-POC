"""
Tool-use ("agentic RAG") variant: instead of always stuffing retrieved
context into the prompt, expose retrieval as a tool. Claude decides whether
and how many times to call it (e.g. multiple searches for a multi-part
question) before answering.

Reuses the VectorStore/Embedder from rag_pipeline.py.
"""

import json
from anthropic import Anthropic
from rag_pipeline import Embedder, VectorStore, MODEL_ID

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the internal knowledge base for information relevant to a query. "
        "Call this whenever you need facts from company docs to answer the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
}


def run_agentic_rag(client: Anthropic, store: VectorStore, user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            system="Answer questions using the search_knowledge_base tool to look up facts before responding.",
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return next(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            results = store.hybrid_search(block.input["query"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(results),
            })
        messages.append({"role": "user", "content": tool_results})


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

    client = Anthropic()

    question = "What's the deploy policy, and who do I contact for IT help?"
    print(f"Q: {question}")
    print(f"A: {run_agentic_rag(client, store, question)}")
