import os
import pathlib

import pytest

from assistant import build_assistant
from fakes import FakeAssistantModel, FakeTriageModel, ToyEmbeddings
from knowledge_base import build_retriever
from tools import check_subscription_status, make_search_docs
from triage import build_triage

DOCS = str(pathlib.Path(__file__).resolve().parent.parent / "tidewave_docs")


@pytest.fixture(scope="module")
def retriever():
    return build_retriever(DOCS, embedding=ToyEmbeddings())


def test_retriever_finds_refund_policy(retriever):
    docs = retriever.invoke("How do refunds work?")
    assert "## Refunds" in docs[0].page_content


def test_search_docs_tool(retriever):
    result = make_search_docs(retriever).invoke({"query": "How do refunds work?"})
    assert "billing@tidewave.example" in result


def test_subscription_tool_normalizes_email():
    assert check_subscription_status.invoke({"customer_email": " Sam@Example.com "}).startswith("Team plan")
    assert check_subscription_status.invoke({"customer_email": "nobody@example.com"}) == "No account found for that email."


def test_triage_flags_urgent_billing():
    result = build_triage(FakeTriageModel()).invoke(
        {"message": "My card was charged twice this month, I need a refund ASAP."}
    )
    assert (result.category, result.priority) == ("billing", "high")


def test_agent_routes_and_remembers(retriever):
    agent = build_assistant(model=FakeAssistantModel(), retriever=retriever)
    config = {"configurable": {"thread_id": "customer-sam"}}
    first = agent.invoke({"messages": [{"role": "user", "content": "How do refunds work?"}]}, config=config)
    second = agent.invoke({"messages": [{"role": "user", "content": "What's my subscription status?"}]}, config=config)

    tools_called = [m.name for m in second["messages"] if m.type == "tool"]
    assert tools_called == ["search_docs", "check_subscription_status"]
    assert len(second["messages"]) == 8
    assert "Team plan" in second["messages"][-1].text


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs a real model")
@pytest.mark.parametrize("question, expected_tool", [
    ("What does the Enterprise plan include?", "search_docs"),
    ("What's the status of my account? I'm sam@example.com", "check_subscription_status"),
])
def test_live_tool_routing(retriever, question, expected_tool):
    agent = build_assistant(retriever=retriever)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": f"routing-{expected_tool}"}},
    )
    assert expected_tool in [m.name for m in result["messages"] if m.type == "tool"]
