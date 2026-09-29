"""HTTP surface tests: auth regression, execution flow, cache lookups, back-pressure."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sandbox_service.api import RequestMetrics, ResultCache, create_app
from sandbox_service.config import Settings
from sandbox_service.exceptions import ServiceBusyError
from sandbox_service.interfaces import ExecutionSpec
from sandbox_service.models import ExecutionRequest, ExecutionResult, ExecutionStatus
from sandbox_service.service import SandboxService
from sandbox_service.testing import FakeBackend

from conftest import _echo_print_handler


def _payload(source: str = "print('hi')", **extra: object) -> dict[str, object]:
    body: dict[str, object] = {"language": "python", "source": source}
    body.update(extra)
    return body


def _authed_client(tmp_path: Path, token: str) -> TestClient:
    """TestClient with bearer auth configured and a fake backend injected."""
    settings = Settings(default_backend="fake", workspace_root=tmp_path / "ws", api_auth_token=token)
    svc = SandboxService(settings, backend=FakeBackend(handler=_echo_print_handler))
    return TestClient(create_app(settings, service=svc))


class TestAuthMiddleware:
    def test_no_auth_configured_allows_anonymous(self, client: TestClient) -> None:
        resp = client.post("/v1/executions", json=_payload())
        assert resp.status_code == 201
        assert resp.json()["stdout"].strip() == "hi"

    def test_missing_token_rejected_with_401(self, tmp_path: Path) -> None:
        with _authed_client(tmp_path, "s3cret") as c:
            resp = c.post("/v1/executions", json=_payload())
            assert resp.status_code == 401
            assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_wrong_token_rejected(self, tmp_path: Path) -> None:
        with _authed_client(tmp_path, "s3cret") as c:
            resp = c.post(
                "/v1/executions", json=_payload(), headers={"Authorization": "Bearer wrong"}
            )
            assert resp.status_code == 401

    def test_correct_token_accepted_and_healthz_public(self, tmp_path: Path) -> None:
        with _authed_client(tmp_path, "s3cret") as c:
            resp = c.post(
                "/v1/executions",
                json=_payload("print(1)"),
                headers={"Authorization": "Bearer s3cret"},
            )
            assert resp.status_code == 201
            assert c.get("/healthz").status_code == 200  # public even with auth on


class TestExecutionEndpoint:
    def test_happy_path_returns_typed_result(self, client: TestClient) -> None:
        resp = client.post("/v1/executions", json=_payload("print(6*7)"))
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "succeeded"
        assert body["stdout"].strip() == "42"
        assert body["backend"] == "fake"

    def test_invalid_payload_is_structured_422(self, client: TestClient) -> None:
        # Regression guard: the old broken auth dependency made *every* POST
        # fail 422 regardless of body; valid bodies must now succeed and only
        # genuinely invalid ones produce pydantic-style errors.
        resp = client.post("/v1/executions", json={"language": "python"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"] == ["body", "source"] for err in detail["details"]["errors"])

    def test_unsupported_language_maps_to_422(self, tmp_path: Path) -> None:
        settings = Settings(
            default_backend="fake", workspace_root=tmp_path / "ws", allowed_languages=["bash"]
        )
        svc = SandboxService(settings, backend=FakeBackend())
        with TestClient(create_app(settings, service=svc)) as c:
            resp = c.post("/v1/executions", json=_payload())
            assert resp.status_code == 422
            assert resp.json()["error"] == "language_not_supported"

    def test_safety_enforcement_rejects_unsafe_code(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/executions",
            json=_payload("import os; os.system('rm -rf /')", enforce_safety=True),
        )
        assert resp.status_code == 422
        assert resp.json()["error"] == "unsafe_code"

    def test_busy_backpressure_raises_service_busy(self) -> None:
        async def scenario() -> None:
            settings = Settings(default_backend="fake", max_concurrent_executions=1)
            gate = asyncio.Event()

            async def blocking_run(spec: ExecutionSpec) -> ExecutionResult:
                await gate.wait()
                return ExecutionResult(
                    execution_id=spec.execution_id, status=ExecutionStatus.SUCCEEDED, exit_code=0
                )

            backend = FakeBackend()
            run_impl: Callable[[ExecutionSpec], Awaitable[ExecutionResult]] = blocking_run
            backend.run = run_impl  # type: ignore[method-assign]
            svc = SandboxService(settings, backend=backend)
            request = ExecutionRequest(language="python", source="print(1)")
            first = asyncio.create_task(svc.execute(request))
            while not svc._semaphore.locked():  # noqa: SLF001 - test synchronization
                await asyncio.sleep(0.001)
            with pytest.raises(ServiceBusyError) as excinfo:
                await svc.execute(request)
            assert excinfo.value.retry_after_seconds > 0
            gate.set()
            await first
            await svc.close()

        asyncio.run(scenario())


class TestResultRetrieval:
    def test_get_by_id_roundtrip(self, client: TestClient) -> None:
        created = client.post("/v1/executions", json=_payload("print('cached')")).json()
        fetched = client.get(f"/v1/executions/{created['execution_id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        resp = client.get("/v1/executions/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"

    def test_cache_evicts_oldest_beyond_bound(self) -> None:
        cache = ResultCache(max_items=2)
        for i in range(3):
            cache.put(
                ExecutionResult(execution_id=f"e{i}", status=ExecutionStatus.SUCCEEDED, exit_code=0)
            )
        assert cache.get("e0") is None
        assert cache.get("e2") is not None
        assert len(cache) == 2

    def test_cache_rejects_nonpositive_bound(self) -> None:
        with pytest.raises(ValueError):
            ResultCache(max_items=0)


class TestDiscoveryEndpoints:
    def test_health_reports_backend_and_circuit(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert body["healthy"] is True
        assert body["backend"] == "fake"

    def test_languages_reflect_settings(self, client: TestClient) -> None:
        assert client.get("/v1/languages").json() == {"languages": ["bash", "python"]}

    def test_metrics_count_executions_and_rejections(self, client: TestClient) -> None:
        client.post("/v1/executions", json=_payload())
        client.post("/v1/executions", json={"bad": 1})
        metrics = client.get("/metrics").json()
        assert metrics["executions"].get("succeeded") == 1
        assert sum(metrics["rejections"].values()) >= 1

