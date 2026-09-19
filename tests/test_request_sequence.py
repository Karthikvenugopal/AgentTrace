from agenttrace.instrumentation.requests import RequestSequenceTracker


def test_sequence_tracks_context_growth_and_inter_request_time() -> None:
    tracker = RequestSequenceTracker()
    first = tracker.observe(prompt_tokens=100, submitted_monotonic=10.0)
    second = tracker.observe(prompt_tokens=145, submitted_monotonic=11.5)
    assert first.sequence_number == 0
    assert first.growth_tokens == 0
    assert first.time_since_previous_request_seconds is None
    assert second.sequence_number == 1
    assert second.growth_tokens == 45
    assert second.time_since_previous_request_seconds == 1.5
