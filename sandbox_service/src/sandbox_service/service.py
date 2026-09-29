"""Application service: orchestrates validation, safety, workspace, execution.

:class:`SandboxService` is the single entry point for all surfaces (Python API,
CLI, HTTP). It owns no kernel itself — everything external arrives through
constructor injection with sane defaults, so swapping backends or making tests
deterministic requires zero patching of internals.

Example:
    >>> import asyncio
    >>> from sandbox_service.service import SandboxService
    >>> from sandbox_service.models import ExecutionRequest
    >>> async def demo() -> None:
    ...     async with SandboxService() as svc:
    ...         result = await svc.execute(ExecutionRequest(
    ...             language="python", source="print('2+2 =', 2+2)"))
    ...         assert result.stdout.strip() == "2+2 = 4"
    >>> asyncio.run(demo())
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import ValidationError

from sandbox_service.adapters.docker import DockerBackend
from sandbox_service.adapters.local import LocalProcessBackend
from sandbox_service.adapters.remote import HttpRemoteBackend
from sandbox_service.adapters.workspace import LocalWorkspaceManager, entrypoint_filename
from sandbox_service.config import Settings
from sandbox_service.exceptions import (
    BackendUnavailableError,
    CircuitOpenError,
    LanguageNotSupportedError,
    SandboxError,
    UnsafeCodeError,
    ValidationError_,
    WorkspaceError,
)
from sandbox_service.interfaces import Clock, ExecutionBackend, ExecutionSpec, WorkspaceManager
from sandbox_service.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    HealthReport,
    ResourceLimits,
    SafetyFinding,
    new_execution_id,
)
from sandbox_service.observability import SystemClock, get_logger
from sandbox_service.safety import SafetyChecker


def _dedupe_findings(findings: list[SafetyFinding]) -> list[SafetyFinding]:
    """De-duplicate safety findings while preserving first-seen order.

    ``SafetyFinding`` is a frozen pydantic model but lists are unhashable, so
    hashing the whole model is unsafe; key on its serialized form instead.

    Args:
        findings: Raw findings from multiple sources (checker + backend).

    Returns:
        A new list with exact duplicates removed.
    """
    seen: set[str] = set()
    out: list[SafetyFinding] = []
    for finding in findings:
        key = finding.model_dump_json()
        if key not in seen:
            seen.add(key)
            out.append(finding)
    return out


def build_backend(
    settings: Settings,
    *,
    fake_records: list[ExecutionResult] | None = None,
) -> ExecutionBackend:
    """Factory mapping ``settings.default_backend`` to a concrete adapter.

    Args:
        settings: Resolved service settings.
        fake_records: Scripted results for the ``fake`` backend (ignored by
            other backends); each :meth:`run` call pops one, and the last
            record repeats once the list is exhausted.

    Returns:
        A fresh backend instance satisfying :class:`ExecutionBackend`.

    Raises:
        BackendUnavailableError: If the selected backend name is unknown or a
            required dependency is missing at construction time (e.g. an
            unconfigured remote URL). Docker/local adapters defer interpreter
            and daemon checks to first use so importing this factory never
            requires those tools.

    Example:
        >>> build_backend(Settings(default_backend="local")).name
        'local'
    """
    name = settings.default_backend
    if name == "local":
        return LocalProcessBackend(settings)
    if name == "docker":
        return DockerBackend(settings)
    if name == "http":
        return HttpRemoteBackend(settings)
    if name == "fake":
        from sandbox_service.testing import FakeBackend

        return FakeBackend(records=list(fake_records or []))
    raise BackendUnavailableError(f"unknown backend {name!r} requested by configuration")


class SandboxService:
    """High-level façade: validate → check safety → prepare workspace → run.

    Attributes:
        settings: The configuration this instance was built with.
        backend_name: Name of the active execution backend.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: ExecutionBackend | None = None,
        workspace: WorkspaceManager | None = None,
        safety: SafetyChecker | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the service with injected dependencies.

        Args:
            settings: Service configuration; defaults to environment-loaded
                :class:`Settings`.
            backend: Execution kernel; defaults to the one selected by
                ``settings.default_backend`` via :func:`build_backend`.
            workspace: Workspace manager; defaults to
                :class:`LocalWorkspaceManager` rooted at the configured path.
            safety: Static checker; defaults to :class:`SafetyChecker`.
            clock: Time source stamped onto results; defaults to system UTC.

        Raises:
            BackendUnavailableError: If default backend construction fails.
        """
        self.settings: Settings = settings or Settings()
        self._backend: ExecutionBackend = backend or build_backend(self.settings)
        self._workspace: WorkspaceManager = workspace or LocalWorkspaceManager(self.settings)
        self._safety: SafetyChecker = safety or SafetyChecker()
        self._clock: Clock = clock or SystemClock()
        self.backend_name: str = self._backend.name
        self._closed = False
        self._log = get_logger(__name__, component="service", backend=self.backend_name)

    # ------------------------------------------------------------------ api
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run one code submission end-to-end and return a typed result.

        Never raises for *execution* problems (timeouts, crashes, backend
        errors become ``ExecutionResult`` values); raises typed exceptions only
        for *request-level* rejections (bad input, unsupported language, unsafe
        code under enforcement).

        Args:
            request: Validated client submission.

        Returns:
            :class:`ExecutionResult` with captured output and status.

        Raises:
            ValidationError_: Request failed domain-model normalization.
            LanguageNotSupportedError: Language outside the allowlist.
            UnsafeCodeError: Blocking safety findings under enforcement
                (per-request flag or ``settings.enforce_safety_by_default``).
            WorkspaceError: Workspace creation/file injection failed.
            CircuitOpenError: Backend breaker rejects the call outright and no
                fallback exists.

        Note:
            Kernel-level failures (backend down after retries, crashes) are
            *not* raised: they come back as ``status=error`` results with a
            machine-readable ``error`` code.
        """
        execution_id = new_execution_id()
        log = self._log.bind(execution_id=execution_id, language=request.language.value)
        limits = self._resolve_limits(request.limits)
        enforce = request.enforce_safety or self.settings.enforce_safety_by_default
        findings = self._check_safety(request, enforce=enforce, log=log)

        workspace_path: Path | None = None
        started = self._clock.monotonic()
        try:
            workspace_path = self._prepare_workspace(execution_id, request)
            await self._write_input_files(workspace_path, request)
            spec = ExecutionSpec(
                execution_id=execution_id,
                language=request.language,
                command=[],  # kernels derive argv from language + entrypoint_name
                source=request.source,
                args=list(request.args),
                stdin=request.stdin,
                env=dict(request.env),
                cwd=workspace_path,
                entrypoint_name=entrypoint_filename(request.language),
                limits=limits,
            )
            result = await self._run_with_fallback(spec, log=log)
        except (CircuitOpenError, UnsafeCodeError, LanguageNotSupportedError, ValidationError_):
            # Request-level policy rejections propagate as typed exceptions;
            # the HTTP/CLI surfaces translate them into structured responses.
            raise
        except SandboxError as exc:
            # Adapters exhausted their own resilience budget (e.g. remote and
            # fallback both unavailable): surface as a typed error result.
            log.error("backend_failed", error=str(exc), code=exc.code)
            result = ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.ERROR,
                error=exc.code,
                stderr=str(exc),
                duration_ms=int((self._clock.monotonic() - started) * 1000),
                backend=self.backend_name,
            )
        except Exception as exc:  # noqa: BLE001 - outermost boundary: never crash the caller
            log.error("execution_crashed", error_type=type(exc).__name__, error=str(exc))
            result = ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.ERROR,
                error=getattr(exc, "code", "internal_error"),
                stderr=f"{type(exc).__name__}: {exc}",
                duration_ms=int((self._clock.monotonic() - started) * 1000),
                backend=self.backend_name,
            )
        finally:
            if workspace_path is not None:
                self._cleanup_workspace(workspace_path, log=log)

        updated = result.model_copy(
            update={
                "created_at": self._clock.now(),
                "findings": _dedupe_findings([*findings, *result.findings]),
            }
        )
        log.info("execute_finished", status=updated.status.value, duration_ms=updated.duration_ms)
        return updated

    async def health(self) -> HealthReport:
        """Aggregate backend and circuit-breaker health.

        Returns:
            A :class:`HealthReport` describing the active kernel.

        Raises:
            Exception: Propagates unexpected probe failures; expected ones are
                reported as ``healthy=False``.
        """
        healthy = await self._backend.health_check()
        circuit = "closed"
        breaker = getattr(self._backend, "breaker", None)
        if breaker is not None:
            circuit = breaker.state
        details: dict[str, Any] = {
            "rlimits_supported": self.settings.supports_rlimits,
            "allowed_languages": sorted(self.settings.allowed_languages),
            "workspace_root": str(self.settings.resolved_workspace_root()),
        }
        return HealthReport(
            healthy=healthy,
            backend=self.backend_name,
            circuit_state=circuit,
            details=details,
        )

    async def close(self) -> None:
        """Close the backend kernel. Idempotent."""
        if self._closed:
            return
        await self._backend.close()
        self._closed = True
        self._log.info("service_closed")

    async def __aenter__(self) -> "SandboxService":
        """Enter the async context, returning the service itself.

        Returns:
            ``self``.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the async context, guaranteeing backend cleanup.

        Args:
            exc_type: In-flight exception type, if any.
            exc: In-flight exception, if any.
            tb: Traceback, if any.
        """
        await self.close()

    # ------------------------------------------------------------- internals
    def _resolve_limits(self, requested: ResourceLimits | None) -> ResourceLimits:
        """Clamp client-supplied limits into the configured envelope.

        Args:
            requested: Optional client limits; ``None`` uses defaults.

        Returns:
            A frozen :class:`ResourceLimits` honoring every ceiling in
            :class:`Settings`.

        Raises:
            ValidationError_: Model bounds violated even after clamping.
        """
        s = self.settings
        if requested is None:
            cpu = min(int(s.max_cpu_seconds), max(1, int(s.default_timeout_s)))
            return ResourceLimits(
                timeout_seconds=s.default_timeout_s,
                memory_mb=min(s.default_memory_mb, s.max_memory_mb),
                max_output_bytes=s.max_output_bytes,
                cpu_seconds=cpu,
                max_file_size_mb=min(s.max_file_size_mb, 512),
                max_processes=min(s.max_processes, 1024),
            )
        timeout = min(requested.timeout_seconds, s.max_timeout_s)
        cpu = min(requested.cpu_seconds, int(timeout) + 1, s.max_cpu_seconds)
        try:
            return ResourceLimits(
                timeout_seconds=timeout,
                memory_mb=min(requested.memory_mb, s.max_memory_mb),
                max_output_bytes=min(requested.max_output_bytes, s.max_output_bytes),
                cpu_seconds=max(1, cpu),
                max_file_size_mb=min(requested.max_file_size_mb, s.max_file_size_mb),
                max_processes=min(requested.max_processes, s.max_processes),
            )
        except ValidationError as exc:
            raise ValidationError_(f"invalid resource limits: {exc.errors()}") from exc

    def _check_safety(
        self,
        request: ExecutionRequest,
        *,
        enforce: bool,
        log: Any,
    ) -> list[SafetyFinding]:
        """Run static analysis and optionally reject blocking findings.

        Args:
            request: Submission to analyze.
            enforce: When True, blocking findings raise instead of annotate.
            log: Bound logger for finding telemetry.

        Returns:
            List of findings collected (possibly empty).

        Raises:
            LanguageNotSupportedError: Language disabled by configuration.
            UnsafeCodeError: Blocking findings present while enforcing policy.
        """
        if request.language.value not in self.settings.allowed_languages:
            raise LanguageNotSupportedError(
                f"language {request.language.value!r} is not enabled",
                allowed=list(self.settings.allowed_languages),
            )
        report = self._safety.inspect(request.language, request.source)
        if report.findings:
            log.warning(
                "safety_findings",
                count=len(report.findings),
                rules=[f.rule for f in report.findings],
            )
        if enforce and not report.allowed:
            raise UnsafeCodeError(
                "submission rejected by safety policy",
                [f.message for f in report.blocking],
            )
        return list(report.findings)

    def _prepare_workspace(self, execution_id: str, request: ExecutionRequest) -> Path:
        """Create the job workspace and write the program entrypoint.

        Directory creation and the small entrypoint write are synchronous
        metadata operations on the injected workspace manager; bulk input-file
        injection happens afterwards in :meth:`_write_input_files`, which is
        async (aiofiles-backed).

        Args:
            execution_id: Unique job id naming the directory.
            request: Submission whose ``source`` becomes the entrypoint file.

        Returns:
            Absolute workspace path.

        Raises:
            WorkspaceError: Directory creation or entrypoint write failed.
        """
        path = self._workspace.create(execution_id)
        entry = path / entrypoint_filename(request.language)
        try:
            entry.write_text(request.source, encoding="utf-8")
        except OSError as exc:
            self._workspace.cleanup(path)
            raise WorkspaceError(f"failed writing entrypoint: {exc}") from exc
        return path

    async def _write_input_files(self, workspace: Path, request: ExecutionRequest) -> None:
        """Inject request.input files asynchronously into the workspace.

        Args:
            workspace: Directory returned by :meth:`_prepare_workspace`.
            request: Submission carrying the file list.

        Raises:
            WorkspaceError: Any file failed containment or I/O checks.
        """
        if request.files:
            await self._workspace.write_files(workspace, request.files)

    async def _run_with_fallback(self, spec: ExecutionSpec, *, log: Any) -> ExecutionResult:
        """Execute on the primary kernel, falling back to local on hard failure.

        Fallback exists so agent workflows degrade gracefully when a remote
        kernel is down; it is logged loudly and marked in the result's
        ``backend`` field.

        Args:
            spec: Resolved job description.
            log: Bound logger.

        Returns:
            Raw backend result (before service-level annotation).

        Raises:
            CircuitOpenError: Primary breaker open AND fallback unavailable.
            BackendUnavailableError: Both kernels failed to accept the job.
        """
        try:
            return await self._backend.run(spec)
        except (BackendUnavailableError, CircuitOpenError) as exc:
            log.warning("primary_backend_failed", error=str(exc), code=getattr(exc, "code", None))
            if isinstance(self._backend, LocalProcessBackend):
                raise
            fallback = LocalProcessBackend(self.settings, clock=self._clock)
            try:
                result = await fallback.run(spec)
            except BackendUnavailableError as inner:
                raise BackendUnavailableError(
                    f"both primary ({exc}) and fallback local backend ({inner}) failed"
                ) from inner
            finally:
                await fallback.close()
            return result.model_copy(update={"stderr": result.stderr + f"\n[fallback from {self.backend_name}: {exc}]"})

    def _cleanup_workspace(self, path: Path, *, log: Any) -> None:
        """Remove a job workspace, logging (not raising) on cleanup failure.

        Cleanup errors must never mask the execution outcome, but they are
        always surfaced in logs — never silently swallowed.

        Args:
            path: Workspace directory.
            log: Bound logger.
        """
        try:
            self._workspace.cleanup(path)
        except WorkspaceError as exc:
            log.error("workspace_cleanup_failed", path=str(path), error=str(exc))
