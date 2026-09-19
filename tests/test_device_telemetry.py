import builtins

from agenttrace.telemetry.device import collect_device_telemetry


def test_device_telemetry_gracefully_handles_missing_torch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    original = builtins.__import__

    def blocked(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    telemetry = collect_device_telemetry()
    assert telemetry.torch_available is False
    assert telemetry.devices == []
    assert "client-process" in telemetry.limitation
