"""Unit tests for pydantic domain models and their boundary validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sandbox_service.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    InputFile,
    Language,
    ResourceLimits,
    SafetyFinding,
    new_execution_id,
)


class TestResourceLimits:
    def test_defaults(self) -> None:
        limits = ResourceLimits()
        assert limits.timeout_seconds == 10.0
        assert limits.memory_mb == 512

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            ResourceLimits().timeout_seconds = 1  # type: ignore[misc]

    def test_cpu_grace_rule(self) -> None:
        with pytest.raises(ValidationError, match="cpu_seconds"):
            ResourceLimits(timeout_seconds=1, cpu_seconds=100)
        assert ResourceLimits(timeout_seconds=10, cpu_seconds=15).cpu_seconds == 15

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResourceLimits(timeout_seconds=0)


class TestInputFile:
    @pytest.mark.parametrize("bad", ["/etc/passwd", "../secret", "..\\win", "a/../../b", "", "nul\x00.txt"])
    def test_unsafe_paths_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            InputFile(path=bad)

    def test_nested_relative_ok(self) -> None:
        assert InputFile(path="data/sub/in.txt", content="x").path == "data/sub/in.txt"


class TestExecutionRequest:
    def test_minimal_valid(self) -> None:
        req = ExecutionRequest(language="python", source="print(1)")
        assert req.language is Language.PYTHON
        assert req.args == [] and req.files == [] and req.limits is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            ExecutionRequest(language="python", source="x", sudo=True)  # type: ignore[call-arg]

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionRequest(language="python", source="")

    def test_null_bytes_in_args_rejected(self) -> None:
        with pytest.raises(ValidationError, match="null bytes"):
            ExecutionRequest(language="python", source="x", args=["ok", "\x00"])

    def test_env_name_validation(self) -> None:
        ExecutionRequest(language="python", source="x", env={"MY_VAR": "1"})
        with pytest.raises(ValidationError, match="invalid environment variable name"):
            ExecutionRequest(language="python", source="x", env={"9bad": "1"})

    def test_unknown_language_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionRequest(language="ruby", source="puts 1")


class TestExecutionResult:
    def test_ok_property_tracks_status(self) -> None:
        ok = ExecutionResult(execution_id="e1", status=ExecutionStatus.SUCCEEDED)
        bad = ExecutionResult(execution_id="e1", status=ExecutionStatus.FAILED, exit_code=1)
        assert ok.ok is True and bad.ok is False

    def test_round_trip_json(self) -> None:
        result = ExecutionResult(
            execution_id="e1",
            status=ExecutionStatus.TIMED_OUT,
            timed_out=True,
            findings=[SafetyFinding(rule="r", message="m", severity="warn")],
        )
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored == result


def test_new_execution_id_unique_and_prefixed() -> None:
    ids = {new_execution_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("exe_") for i in ids)
