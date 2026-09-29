"""HTTP API surface: FastAPI app factory exposing the service over REST.

Endpoints (all JSON):

* ``GET  /healthz``          — liveness + backend/circuit status
* ``POST /v1/executions``    — run a submission, return :class:`ExecutionResult`
* ``GET  /v1/languages``     — configured language allowlist
* ``GET  /metrics``          — lightweight request counters (JSON)

The application service is created inside the app's lifespan and exposed via
``app.state``; tests build the app with an injected service through
:func:`create_app` — no global state, no module-level singletons.

Auth: when ``Settings.api_auth_token`` is set, every route except ``/healthz``
requires ``Authorization: Bearer <token>``.

Example:
    >>> from fastapi.testclient import TestClient
    >>> from sandbox_service.api import create_app
    >>> client = TestClient(create_app())   # doctest: +SKIP
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from sandbox_service.config import Settings
from sandbox_service.exceptions import (
    CircuitOpenError,
    LanguageNotSupportedError,
    SandboxError,
    UnsafeCodeError,
    ValidationError_,
)
from sandbox_service.models import ExecutionRequest, ExecutionResult, HealthReport
from sandbox_service.observability import get_logger
from sandbox_service.service import SandboxService, build_backend


class RequestMetrics:
    """Thread-safe counters for observability without external dependencies.

    Counters are keyed by terminal execution status and by machine-readable
    rejection code respectively; :meth:`snapshot` aggregates them.
    """

    def __init__(self) -> None:
        """Initialize zeroed counters."""
        self._lock = threading.Lock()
        self._executions: dict[str, int] = {}
        self._rejections: dict[str, int] = {}

    def record_execution(self, status_value: str) -> None:
        """Count one finished execution by its terminal status.

        Args:
            status_value: ``ExecutionStatus`` value string.
        """
        with self._lock:
            self._executions[status_value] = self._executions.get(status_value, 0) + 1

    def record_rejection(self, code: str) -> None:
        """Count one rejected request by machine-readable error code.

        Args:
            code: The ``SandboxError.code`` (or ``"invalid_request"``).
        """
        with self._lock:
            self._rejections[code] = self._rejections.get(code, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of all counters.

        Returns:
            Dict with ``executions``, ``rejections`` and ``total_requests``.
        """
        with self._lock:
            total = sum(self._executions.values()) + sum(self._rejections.values())
            return {
                "total_requests": total,
                "executions": dict(self._executions),
                "rejections": dict(self._rejections),
            }


def create_app(
    settings: Settings | None = None,
    *,
    service: SandboxService | None = None,
) -> FastAPI:
    """Build the FastAPI application (dependency-injectable factory).

    Args:
        settings: Configuration source; defaults to environment-loaded
            :class:`~sandbox_service.config.Settings`. Ignored when
            ``service`` is provided.
        service: Pre-built service (used by tests to inject fakes). When
            omitted, one is constructed in the app lifespan from
            :func:`~sandbox_service.service.build_backend`.

    Returns:
        A configured :class:`fastapi.FastAPI` instance. Its lifespan owns
        service startup/shutdown; use as a context manager or under uvicorn.

    Example:
        >>> from sandbox_service.testing import FakeBackend
        >>> from sandbox_service.service import SandboxService
        >>> svc = SandboxService(backend=FakeBackend())
        >>> app = create_app(service=svc)
        >>> [route.path for route in app.routes if hasattr(route, "path")][:4]
        ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc']
    """
    resolved_settings = settings or (service.settings if service else Settings())
    log = get_logger(__name__, component="http_api")
    metrics = RequestMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Create and dispose the service around the app lifetime.

        Args:
            app: The application whose ``state`` receives the service.

        Yields:
            Control back to the server while the service is live.
        """
        own_service = service is None
        active = service or SandboxService(
            resolved_settings, backend=build_backend(resolved_settings)
        )
        app.state.service = active
        app.state.settings = resolved_settings
        app.state.metrics = metrics
        log.info("api_startup", backend=active.backend_name)
        try:
            yield
        finally:
            if own_service:
                await active.close()
            log.info("api_shutdown")

    app = FastAPI(
        title="sandbox-service",
        version="0.1.0",
        description="Local-first sandboxed code execution for agent stacks.",
        lifespan=lifespan,
    )
    # auto_error=False is essential: unauthenticated requests must reach our
    # logic (which no-ops when auth is disabled) instead of a generic 403.
    bearer = HTTPBearer(auto_error=False)

    async def require_token(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        """Reject requests lacking the configured bearer token.

        Args:
            credentials: Parsed Authorization header, if present.

        Raises:
            HTTPException: 401 when auth is configured and the token is
                missing or wrong.
        """
        expected = resolved_settings.api_auth_token
        if not expected:
            return
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "unauthorized", "message": "bearer token required"},
            )
        if credentials.credentials != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "unauthorized", "message": "invalid token"},
            )

    @app.exception_handler(SandboxError)
    async def sandbox_error_handler(request: Request, exc: SandboxError) -> Response:
        """Map typed domain errors onto stable HTTP codes/JSON bodies.

        Args:
            request: In-flight request (used for the logged path).
            exc: The domain error raised inside a route.

        Returns:
            JSON response with ``error``, ``message`` and ``details`` keys.
        """
        metrics.record_rejection(exc.code)
        log.warning("request_rejected", path=str(request.url.path), code=exc.code)
        code_map: dict[str, int] = {
            ValidationError_.code: status.HTTP_422_UNPROCESSABLE_ENTITY,
            UnsafeCodeError.code: status.HTTP_422_UNPROCESSABLE_ENTITY,
            LanguageNotSupportedError.code: status.HTTP_422_UNPROCESSABLE_ENTITY,
            CircuitOpenError.code: status.HTTP_503_SERVICE_UNAVAILABLE,
        }
        http_code = code_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        headers: dict[str, str] = {}
        if isinstance(exc, CircuitOpenError):
            headers["Retry-After"] = str(int(exc.retry_after_seconds) + 1)
        return Response(
            content=_json_dumps(exc.to_dict()),
            status_code=http_code,
            media_type="application/json",
            headers=headers,
        )

    @app.get("/healthz", response_model=HealthReport)
    async def healthz() -> HealthReport:
        """Liveness/readiness probe including breaker state.

        Returns:
            Aggregated :class:`~sandbox_service.models.HealthReport`.
        """
        svc: SandboxService = app.state.service
        return await svc.health()

    @app.get("/v1/languages")
    async def languages() -> dict[str, list[str]]:
        """Advertise the configured language allowlist.

        Returns:
            ``{"languages": [...]}`` sorted alphabetically.
        """
        return {"languages": sorted(resolved_settings.allowed_languages)}

    @app.get("/metrics")
    async def metrics_endpoint() -> dict[str, Any]:
        """Expose in-process request counters as JSON.

        Returns:
            Snapshot from :meth:`RequestMetrics.snapshot`.
        """
        return metrics.snapshot()

    @app.post(
        "/v1/executions",
        response_model=ExecutionResult,
        dependencies=[Depends(require_token)],
    )
    async def create_execution(payload: dict[str, Any]) -> ExecutionResult:
        """Run one submission end-to-end.

        Args:
            payload: JSON body matching
                :class:`~sandbox_service.models.ExecutionRequest`.

        Returns:
            The typed execution result (captured stdout/stderr/status).

        Raises:
            HTTPException: 422 with a structured body when the payload fails
                model validation; typed domain errors are converted by the
                app-level exception handler.
        """
        svc: SandboxService = app.state.service
        try:
            request = ExecutionRequest.model_validate(payload)
        except ValidationError as exc:
            metrics.record_rejection(ValidationError_.code)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": ValidationError_.code,
                    "message": "execution request failed validation",
                    "details": {"errors": _safe_errors(exc)},
                },
            ) from exc
        result = await svc.execute(request)
        metrics.record_execution(result.status.value)
        return result

    return app


def _json_dumps(obj: object) -> str:
    """Serialize a small dict to compact JSON.

    Args:
        obj: JSON-compatible object.

    Returns:
        Compact JSON string (non-serializable values are stringified).
    """
    return json.dumps(obj, default=str)


def _safe_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Reduce a pydantic ValidationError to JSON-safe entries.

    Args:
        exc: The validation error to summarize.

    Returns:
        List of ``{loc, msg, type}`` dicts with non-serializable values
        stringified.
    """
    out: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False):
        entry: dict[str, Any] = {
            "loc": list(err.get("loc", [])),
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        out.append(entry)
    return out

