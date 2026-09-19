from agenttrace.instrumentation.timing import (
    expanded_token_arrival_intervals,
    inter_chunk_intervals,
)
from agenttrace.serving.client import StreamObservation


def test_chunk_intervals_are_not_described_as_token_latency() -> None:
    chunks = [
        StreamObservation(elapsed_seconds=0.1, text="one", estimated_tokens=1),
        StreamObservation(elapsed_seconds=0.25, text="two three", estimated_tokens=2),
    ]
    assert inter_chunk_intervals(chunks) == [0.15]
    assert expanded_token_arrival_intervals(chunks) == [0.1, 0.15, None]
