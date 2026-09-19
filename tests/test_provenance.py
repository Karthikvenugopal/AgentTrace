from agenttrace.benchmarking.provenance import capture_provenance


def test_provenance_contains_versions_but_not_environment() -> None:
    provenance = capture_provenance()
    assert provenance["python_version"]
    assert "pydantic" in provenance["dependencies"]
    assert "environment" not in provenance
    assert "device" in provenance
