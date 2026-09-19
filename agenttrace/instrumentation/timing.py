"""Clock-safe latency and concurrency measurements."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise

from agenttrace.serving.client import StreamObservation


@dataclass(frozen=True)
class ClockSnapshot:
    wall_time: datetime
    monotonic: float


def snapshot() -> ClockSnapshot:
    return ClockSnapshot(wall_time=datetime.now(UTC), monotonic=time.monotonic())


def elapsed(start: ClockSnapshot, end: ClockSnapshot) -> float:
    """Use the monotonic clock for duration, never wall-clock subtraction."""

    return max(0.0, end.monotonic - start.monotonic)


def inter_chunk_intervals(chunks: list[StreamObservation]) -> list[float]:
    """Return observed stream arrival intervals.

    These are chunk arrival intervals, not decoding latency. One chunk may contain
    multiple tokens and transports can buffer chunks.
    """

    return [
        current.elapsed_seconds - previous.elapsed_seconds for previous, current in pairwise(chunks)
    ]


def expanded_token_arrival_intervals(chunks: list[StreamObservation]) -> list[float | None]:
    """Represent token timing without pretending multi-token chunks were decoded together.

    The first token in a chunk gets the observed interval; additional tokens have
    unknown intervals and are represented as ``None``.
    """

    values: list[float | None] = []
    prior = 0.0
    for chunk in chunks:
        interval = chunk.elapsed_seconds - prior
        count = max(1, chunk.estimated_tokens)
        values.append(interval)
        values.extend([None] * (count - 1))
        prior = chunk.elapsed_seconds
    return values


class ActiveRequestCounter:
    """Async-safe gauge returning concurrency including the entering request."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @asynccontextmanager
    async def track(self) -> AsyncIterator[int]:
        async with self._lock:
            self._active += 1
            current = self._active
        try:
            yield current
        finally:
            async with self._lock:
                self._active -= 1
