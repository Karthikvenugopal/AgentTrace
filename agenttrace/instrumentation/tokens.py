"""Tokenizer abstraction and role-aware prompt accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from agenttrace.models import ChatMessage, MessageRole
from agenttrace.tracing.schema import PromptTokenBreakdown


class TokenCounter(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def method(self) -> str: ...

    def count(self, text: str) -> int: ...

    def construct_text(self, target_tokens: int, seed_text: str = "agent trace") -> str: ...


@dataclass(frozen=True)
class WhitespaceTokenCounter:
    """Deterministic fallback for tests; not a model-tokenizer substitute."""

    identity: str = "agenttrace/whitespace-v1"
    method: str = "whitespace_estimate"

    def count(self, text: str) -> int:
        return len(text.split())

    def construct_text(self, target_tokens: int, seed_text: str = "agent trace") -> str:
        if target_tokens < 0:
            raise ValueError("target_tokens cannot be negative")
        vocabulary = seed_text.split() or ["token"]
        return " ".join(vocabulary[index % len(vocabulary)] for index in range(target_tokens))


class TransformersTokenCounter:
    """Exact counter backed by a Hugging Face tokenizer loaded on demand."""

    method = "transformers_encode"

    def __init__(self, model_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install agenttrace[tokenizers] for exact token counting") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        self._identity = model_or_path

    @property
    def identity(self) -> str:
        return self._identity

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def construct_text(self, target_tokens: int, seed_text: str = "agent trace") -> str:
        if target_tokens < 0:
            raise ValueError("target_tokens cannot be negative")
        seed_ids = self._tokenizer.encode(seed_text, add_special_tokens=False)
        if not seed_ids:
            raise ValueError("seed_text does not produce tokens")
        ids = [seed_ids[index % len(seed_ids)] for index in range(target_tokens)]
        text = self._tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        actual = self._tokenizer.encode(text, add_special_tokens=False)
        if len(actual) != target_tokens:
            # Tokenizer decode/encode is not always stable. Truncate/pad using a known ID.
            actual = actual[:target_tokens]
            actual.extend([seed_ids[0]] * (target_tokens - len(actual)))
            text = self._tokenizer.decode(actual, clean_up_tokenization_spaces=False)
        return cast(str, text)


@dataclass(frozen=True)
class TokenAccounting:
    total: int
    breakdown: PromptTokenBreakdown


def count_messages(messages: list[ChatMessage], counter: TokenCounter) -> TokenAccounting:
    counts = {role: 0 for role in MessageRole}
    for message in messages:
        counts[message.role] += counter.count(message.content)
    breakdown = PromptTokenBreakdown(
        system=counts[MessageRole.SYSTEM],
        user=counts[MessageRole.USER],
        assistant=counts[MessageRole.ASSISTANT],
        tool=counts[MessageRole.TOOL],
    )
    return TokenAccounting(total=breakdown.total, breakdown=breakdown)
