from agenttrace.instrumentation.tokens import (
    WhitespaceTokenCounter,
    build_token_counter,
    count_messages,
)
from agenttrace.models import ChatMessage, MessageRole


def test_role_aware_token_accounting() -> None:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="one two"),
        ChatMessage(role=MessageRole.USER, content="three"),
        ChatMessage(role=MessageRole.ASSISTANT, content="four five six"),
        ChatMessage(role=MessageRole.TOOL, content="seven eight"),
    ]
    accounting = count_messages(messages, WhitespaceTokenCounter())
    assert accounting.total == 8
    assert accounting.breakdown.model_dump() == {
        "system": 2,
        "user": 1,
        "assistant": 3,
        "tool": 2,
    }


def test_constructed_prompt_has_requested_length() -> None:
    counter = WhitespaceTokenCounter()
    text = counter.construct_text(257, "alpha beta gamma")
    assert counter.count(text) == 257


def test_unconfigured_counter_is_explicitly_an_estimate() -> None:
    counter = build_token_counter(None)
    assert counter.identity == "agenttrace/whitespace-v1"
    assert counter.method == "whitespace_estimate"
