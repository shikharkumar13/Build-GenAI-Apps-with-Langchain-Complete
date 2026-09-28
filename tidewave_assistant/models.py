from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def get_model(
    provider_and_model: str = "anthropic:claude-sonnet-5", **kwargs
) -> BaseChatModel:
    """Return a configured chat model for Tidewave's assistant.

    Accepts an init_chat_model-style "provider:model-name" string, so
    swapping the underlying provider anywhere in the project later is a
    one-argument change, not a rewrite. Any extra settings, such as
    reasoning_effort="low", are passed straight through.
    """
    return init_chat_model(provider_and_model, **kwargs)
