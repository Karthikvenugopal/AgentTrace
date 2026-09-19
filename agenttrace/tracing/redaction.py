"""Prompt-content privacy policies for trace persistence."""

from __future__ import annotations

import re
from typing import Literal

from agenttrace.models import ChatMessage


def redact_text(text: str, patterns: list[str], *, redact_all: bool) -> str:
    if redact_all and not patterns:
        return "[REDACTED]"
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "[REDACTED]", result)
    return result


def content_fields(
    messages: list[ChatMessage],
    *,
    mode: Literal["full", "redacted", "metadata_only"],
    patterns: list[str],
) -> dict[str, object]:
    if mode == "metadata_only":
        return {"prompt": None, "messages": None}
    redact_all = mode == "redacted"
    sanitized = [
        message.model_copy(
            update={"content": redact_text(message.content, patterns, redact_all=redact_all)}
        )
        for message in messages
    ]
    return {
        "prompt": "\n".join(message.content for message in sanitized),
        "messages": [message.model_dump(mode="json", exclude_none=True) for message in sanitized],
    }
