from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from knowledge_base import build_retriever
from models import get_model
from tools import check_subscription_status, make_search_docs

SYSTEM_PROMPT = (
    "You are a helpful assistant for Tidewave. Use search_docs for "
    "questions about features, plans, or policies. Use "
    "check_subscription_status for questions about a specific "
    "customer's account; ask for their email if you don't have it."
)


def build_assistant(
    model: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
    checkpointer=None,
):
    model = model or get_model()
    tools = [make_search_docs(retriever or build_retriever()), check_subscription_status]
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=6),
            SummarizationMiddleware(model=model, trigger=("messages", 40), keep=("messages", 20)),
        ],
        checkpointer=checkpointer or InMemorySaver(),
        name="tidewave_assistant",
    )
