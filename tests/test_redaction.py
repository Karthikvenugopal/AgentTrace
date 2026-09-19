from agenttrace.models import ChatMessage, MessageRole
from agenttrace.tracing.redaction import content_fields


def test_metadata_mode_stores_no_content() -> None:
    result = content_fields(
        [ChatMessage(role=MessageRole.USER, content="secret")],
        mode="metadata_only",
        patterns=[],
    )
    assert result == {"prompt": None, "messages": None}


def test_redacted_mode_defaults_to_removing_all_content() -> None:
    result = content_fields(
        [ChatMessage(role=MessageRole.USER, content="private repository text")],
        mode="redacted",
        patterns=[],
    )
    assert result["prompt"] == "[REDACTED]"


def test_full_mode_can_still_apply_secret_patterns() -> None:
    result = content_fields(
        [ChatMessage(role=MessageRole.USER, content="token=sk-sensitive safe")],
        mode="full",
        patterns=[r"sk-[a-z]+"],
    )
    assert "sk-sensitive" not in str(result)
    assert "safe" in str(result)
