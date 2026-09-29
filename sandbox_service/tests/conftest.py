"""Shared fixtures: hermetic settings, deterministic fakes, HTTP test client."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sandbox_service.api import create_app
from sandbox_service.config import Settings
from sandbox_service.interfaces import Clock, ExecutionSpec
from sandbox_service.models import ExecutionResult, ExecutionStatus
from sandbox_service.service import SandboxService
from sandbox_service.testing import FakeBackend


class FakeClock:
    """Monotonic-only fake clock: ``monotonic()`` advances per tick, ``now()`` is fixed.

    Deterministic by construction — no reliance on wall time anywhere.
    """

    def __init__(self, start: float = 1000.0, step: float = 0.5) -> None:
        self._value = start
        self._step = step
        self.now_calls = 0

    def now(self) -> datetime:
        self.now_calls += 1
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.now_calls)

    def monotonic(self) -> float:
        value = self._value
        self._value += self._step
        return value


class InstantSleeper:
    """Sleeper port that records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


class SeededRng:
    """RandomSource returning a constant midpoint (jitter-free, seedable)."""

    def __init__(self, factor: float = 0.75) -> None:
        self._factor = factor

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._factor


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip SANDBOX_* env vars so ambient configuration can't leak into tests."""
    for key in [k for k in os.environ if k.startswith("SANDBOX_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings rooted at a throwaway workspace, fake backend by default."""
    return Settings(default_backend="fake", workspace_root=tmp_path / "ws", log_level="WARNING")


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_backend() -> FakeBackend:
    """Plain echo fake backend (raw source on stdout)."""
    return FakeBackend()


def _echo_print_handler(spec: ExecutionSpec) -> ExecutionResult:
    """Emulate CPython ``print(<expr>)`` for constant expressions safely.

    Uses :func:`ast.parse` + :func:`ast.literal_eval` on the folded tree
    (never ``eval``/``compile``), so only literal and constant-foldable
    arithmetic arguments are interpreted; anything else echoes the source.

    Args:
        spec: Resolved execution description.

    Returns:
        Successful result whose stdout mirrors what CPython would print for
        constant-expression ``print(...)`` sources; echoes the source
        otherwise.
    """
    source = spec.source.strip()
    stdout = source
    if source.startswith("print(") and source.endswith(")"):
        inner = source[len("print(") : -1].strip()
        try:
            stdout = str(ast.literal_eval(ast.parse(inner, mode="eval", feature_version=(3, 12))))
        except (ValueError, SyntaxError):
            stdout = inner.strip("'\"")
    return ExecutionResult(
        execution_id=spec.execution_id,
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        stdout=f"{stdout}\n",
        backend="fake",
    )


@pytest.fixture
def printing_backend() -> FakeBackend:
    """Fake backend that "executes" ``print(<literal>)`` like a real kernel."""
    return FakeBackend(handler=_echo_print_handler)


@pytest.fixture
def service(tmp_settings: Settings, fake_backend: FakeBackend, fake_clock: FakeClock) -> SandboxService:
    svc = SandboxService(tmp_settings, backend=fake_backend, clock=fake_clock)
    yield svc


@pytest.fixture
def succeeded_result() -> ExecutionResult:
    return ExecutionResult(execution_id="", status=ExecutionStatus.SUCCEEDED, exit_code=0, stdout="ok\n")


@pytest.fixture
def client(
    tmp_settings: Settings, printing_backend: FakeBackend, fake_clock: FakeClock
) -> Iterator[TestClient]:
    """TestClient wired to an injected service; lifespan runs via context mgr."""
    svc = SandboxService(tmp_settings, backend=printing_backend, clock=fake_clock)
    app = create_app(tmp_settings, service=svc)
    with TestClient(app) as test_client:
        yield test_client
