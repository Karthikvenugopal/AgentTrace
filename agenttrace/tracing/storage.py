"""Streaming JSONL trace persistence with crash-tolerant flushes."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import TextIO

from agenttrace.tracing.schema import AnyTraceRecord, TraceHeader
from agenttrace.tracing.validation import parse_record


class TraceWriter:
    """Thread-safe append-only writer for long-running agent experiments."""

    def __init__(self, path: Path, *, flush_each_record: bool = True) -> None:
        self.path = path
        self.flush_each_record = flush_each_record
        self._handle: TextIO | None = None
        self._lock = threading.Lock()
        self._records_written = 0

    @property
    def records_written(self) -> int:
        return self._records_written

    def open(self) -> TraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def write(self, record: AnyTraceRecord) -> None:
        if self._handle is None:
            raise RuntimeError("TraceWriter must be opened before writing")
        encoded = record.model_dump_json(exclude_none=True)
        with self._lock:
            self._handle.write(encoded + "\n")
            self._records_written += 1
            if self.flush_each_record:
                self._handle.flush()
                os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def __enter__(self) -> TraceWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def iter_trace(path: Path) -> Iterator[AnyTraceRecord]:
    """Lazily parse a trace without holding the complete experiment in memory."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield parse_record(line)
            except Exception as exc:
                raise ValueError(f"invalid trace record at {path}:{line_number}: {exc}") from exc


def read_header(path: Path) -> TraceHeader:
    try:
        first = next(iter_trace(path))
    except StopIteration as exc:
        raise ValueError(f"trace is empty: {path}") from exc
    if not isinstance(first, TraceHeader):
        raise ValueError(f"first trace record is not a header: {path}")
    return first


def write_manifest(path: Path, values: dict[str, object]) -> None:
    """Atomically write a replay manifest next to trace artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
