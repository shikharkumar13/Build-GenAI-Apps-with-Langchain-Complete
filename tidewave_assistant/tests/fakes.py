"""Stand-ins that let the whole project run with no API keys.

Each one replaces exactly one external decision: which tool to call,
how to classify a message, or what a text's embedding vector is.
Everything else in the project runs unmodified."""
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from triage import TicketClassification

VOCAB = ["plan", "bill", "refund", "team", "enterprise", "project", "slack"]


class ToyEmbeddings(Embeddings):
    """Article 10's normalized bag-of-words vectorizer."""

    def _vectorize(self, text: str) -> list[float]:
        words = [w.rstrip("s") if w.endswith("s") and len(w) > 3 else w
                 for w in re.findall(r"[a-z]+", text.lower())]
        counts = Counter(words)
        vec = [float(counts.get(term, 0)) for term in VOCAB]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


class FakeAssistantModel(BaseChatModel):
    """Picks a tool by keyword in the latest question, then answers with
    the tool's result. A real model reasons about which tool to call;
    this picks deterministically so the run is reproducible. Only the
    most recent message is checked for a tool result, so a stale one
    from an earlier turn in the same thread isn't mistaken for a fresh
    answer."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content=f"Here's what I found: {messages[-1].content}")
            return ChatResult(generations=[ChatGeneration(message=message)])

        turn = sum(isinstance(m, HumanMessage) for m in messages)
        question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        if "subscription" in question.lower() or "status" in question.lower():
            name, args = "check_subscription_status", {"customer_email": "sam@example.com"}
        else:
            name, args = "search_docs", {"query": question}
        tool_call = {"name": name, "args": args, "id": f"call_{turn}", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-assistant-model"


def _classify(prompt_value) -> TicketClassification:
    message = prompt_value.to_messages()[-1].content.lower()
    urgent = any(w in message for w in ("asap", "immediately", "urgent"))
    is_billing = any(w in message for w in ("charge", "refund", "billing", "payment"))
    return TicketClassification(
        category="billing" if is_billing else "other",
        priority="high" if (urgent and is_billing) else "low",
        summary=message[:60],
    )


class FakeTriageModel(BaseChatModel):
    """Keyword matching in place of a real model reading the message."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("only used through with_structured_output")

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(_classify)

    @property
    def _llm_type(self) -> str:
        return "fake-triage-model"
