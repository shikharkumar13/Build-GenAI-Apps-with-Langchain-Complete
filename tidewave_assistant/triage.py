from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from models import get_model


class TicketClassification(BaseModel):
    """Structured classification of a Tidewave support message."""
    category: Literal["billing", "technical", "feature_request", "other"] = Field(
        description="The best-fit category for this support message"
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="How urgently this message needs a response"
    )
    summary: str = Field(description="One-sentence summary of the issue")


classify_template = ChatPromptTemplate.from_messages([
    ("system", "You classify incoming support messages for Tidewave."),
    ("human", "{message}"),
])


def build_triage(model: BaseChatModel | None = None) -> Runnable:
    classifier = (model or get_model()).with_structured_output(
        TicketClassification, method="json_schema"
    )
    return classify_template | classifier
