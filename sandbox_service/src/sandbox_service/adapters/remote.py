"""Remote HTTP execution backend: call out to another sandbox-service instance.

Demonstrates the provider-agnostic port for deployments where the kernel runs
on another machine (e.g. a GPU box or a dedicated VM). Uses httpx with explicit
timeouts and is wrapped in retry + circuit breaking. The client is injectable,
so tests drive it through ``httpx.MockTransport`` without any network.

Satisfies :class:`~sandbox_service.interfaces.ExecutionBackend`.

Example:
    >>> from sandbox_service.config import Settings
    >>> b = HttpRemoteBackend(Settings(remote_base_url="http://kernel:8090"))
    >>> b.name
    'remote-http'
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx

from sandbox_service.config import Settings
from sandbox_service.exceptions import BackendUnavailableError, CircuitOpenError
from sandbox_service.interfaces import Clock, ExecutionSpec, RandomSource, Sleeper
from sandbox_service.models import ExecutionResult, ExecutionStatus
from sandbox_service.observability import AsyncSleeper, SeededRandom, SystemClock, get_logger
from sandbox_service.resilience import CircuitBreaker, RetryPolicy


class HttpRemoteBackend:
    """Posts executions to a remote ``/v1/executions`` endpoint.

    Attributes:
        name: Always ``"remote-http"``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
        sleeper: Sleeper | None = None,
        rng: RandomSource | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            settings: Remote URL/key plus retry/breaker/timeout configuration.
            client_factory: Injectable factory returning an ``AsyncClient``
                (pass one backed by ``httpx.MockTransport`` in tests).
            sleeper: Sleep port used by retries.
            rng: Seedable randomness source for jitter.
            clock: Duration-measurement clock.

        Raises:
            ConfigurationError: If ``remote_base_url`` is empty.
        """
        if not settings.remote_base_url:
            raise BackendUnavailableError("SANDBOX_REMOTE_BASE_URL is not configured")
        self.name: str = "remote-http"
        self._settings = settings
        self._clock: Clock = clock or SystemClock()
        self._sleeper: Sleeper = sleeper or AsyncSleeper()
        self._rng: RandomSource = rng or SeededRandom(seed=20260929)
        headers = {"User-Agent": "sandbox-service/0.1"}
        if settings.remote_api_key:
            headers["Authorization"] = f"Bearer {settings.remote_api_key}"
        self._headers = headers
        self._client_factory = client_factory or self._default_client_factory
        self._client: httpx.AsyncClient | None = None
        self._breaker = CircuitBreaker(
            threshold=settings.circuit_failure_threshold,
            recovery_seconds=settings.circuit_recovery_timeout_s,
            clock=self._clock,
            name="remote-http",
        )
        self._retry = RetryPolicy[ExecutionResult](
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_s,
            max_delay=settings.retry_max_delay_s,
            sleeper=self._sleeper,
            rng=self._rng,
            breaker=self._breaker,
            retry_on=(httpx.TransportError, asyncio.TimeoutError),
        )
        self._log = get_logger(__name__, component="backend", backend=self.name)

    def _default_client_factory(self, **kwargs: object) -> httpx.AsyncClient:
        """Build the production httpx client.

        Args:
            **kwargs: Passed through to :class:`httpx.AsyncClient`.

        Returns:
            A new async client with sane connection limits.
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.remote_request_timeout_s),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            **kwargs,  # type: ignore[arg-type]
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily construct the shared client.

        Returns:
            The cached :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            self._client = self._client_factory(headers=dict(self._headers))
        return self._client

    @property
    def breaker(self) -> CircuitBreaker:
        """Circuit breaker guarding the remote dependency.

        Returns:
            The breaker instance for health reporting.
        """
        return self._breaker

    async def health_check(self) -> bool:
        """GET ``/healthz`` on the remote service.

        Returns:
            True when the remote answers 200 with ``healthy: true``.
        """
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._settings.remote_base_url.rstrip('/')}/healthz",
                timeout=5.0,
            )
            return resp.status_code == 200 and bool(resp.json().get("healthy"))
        except (httpx.HTTPError, ValueError):
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client. Idempotent."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Submit one job to the remote service and parse its typed result.

        Args:
            spec: Resolved execution description; the remote side re-validates
                and enforces its own limit ceilings.

        Returns:
            Parsed :class:`ExecutionResult` tagged with this backend's name.

        Raises:
            CircuitOpenError: If the breaker rejects the call outright.
            BackendUnavailableError: On transport failure after retries, or a
                non-2xx response that isn't a structured rejection (422).
        """
        payload = {
            "execution_id": spec.execution_id,
            "language": spec.language.value,
            "source": spec.source,
            "args": spec.args,
            "stdin": spec.stdin,
            "env": spec.env,
            "limits": spec.limits.model_dump(),
        }
        url = f"{self._settings.remote_base_url.rstrip('/')}/v1/executions"

        async def _attempt() -> ExecutionResult:
            client = await self._get_client()
            resp = await client.post(url, json=payload)
            if resp.status_code in (400, 422):
                # Structured rejection from the remote validator: a real answer,
                # not a transport failure — do NOT retry or trip the breaker.
                detail = resp.json()
                return ExecutionResult(
                    execution_id=spec.execution_id,
                    status=ExecutionStatus.REJECTED,
                    error=str(detail.get("error", "remote_rejected")),
                    stderr=str(detail.get("message", "")),
                    backend=self.name,
                )
            resp.raise_for_status()
            result = ExecutionResult.model_validate(resp.json())
            return result.model_copy(update={"backend": self.name})

        try:
            return await self._retry.execute(_attempt, description=f"remote run {spec.execution_id}")
        except CircuitOpenError:
            raise
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            self._breaker.record_failure()
            self._log.error("remote_unavailable", error=str(exc))
            raise BackendUnavailableError(
                f"remote sandbox service failed: {exc}",
                details={"url": url, "attempts": self._settings.retry_max_attempts},
            ) from exc
