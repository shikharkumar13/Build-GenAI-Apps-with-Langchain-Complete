from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from knowledge_base import format_docs


def make_search_docs(retriever: BaseRetriever) -> BaseTool:
    docs_chain = retriever | format_docs

    @tool
    def search_docs(query: str) -> str:
        """Search Tidewave's documentation for information about features, plans, or policies."""
        return docs_chain.invoke(query)

    return search_docs


@tool
def check_subscription_status(customer_email: str) -> str:
    """Check a Tidewave customer's current subscription plan and billing status."""
    fake_db = {
        "sam@example.com": "Team plan, active, next billing date 2026-09-15",
        "priya@example.com": "Starter plan, active",
    }
    return fake_db.get(customer_email.strip().lower(), "No account found for that email.")
