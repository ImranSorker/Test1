"""Docker execution backend: strongest isolation when a daemon is available.

Each job runs in a fresh, ephemeral container built from the configured image:
``--network none``, read-only rootfs, dropped capabilities, non-root user,
pids/memory caps, and the per-job workspace bind-mounted read-write at ``/work``.
The CLI invocation itself is wrapped with retry + circuit breaking because the
docker daemon is an external dependency.

Satisfies :class:`~sandbox_service.interfaces.ExecutionBackend`.

Example:
    >>> from sandbox_service.config import Settings
    >>> backend = DockerBackend(Settings(docker_cli_path="/nonexistent/docker"))
    >>> backend.name
    'docker'
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import shutil
from collections.abc import Sequence

from sandbox_service.config import Settings
from sandbox_service.exceptions import BackendUnavailableError
from sandbox_service.interfaces import Clock, ExecutionSpec, RandomSource, Sleeper
from sandbox_service.models import ExecutionResult, ExecutionStatus
from sandbox_service.observability import AsyncSleeper, SeededRandom, SystemClock, get_logger
from sandbox_service.resilience import CircuitBreaker, RetryPolicy


class DockerBackend:
    """Runs each snippet inside a throwaway hardened container.

    Attributes:
        name: Always ``"docker"``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cli_path: str | None = None,
        sleeper: Sleeper | None = None,
        rng: RandomSource | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            settings: Image name, container prefix, limit ceilings, retry and
                breaker configuration.
            cli_path: Override for the docker executable path (injected in
                tests); defaults to ``settings.docker_cli_path`` on PATH.
            sleeper: Sleep port used by retries (fake it in tests).
            rng: Randomness source for jitter (seedable).
            clock: Duration-measurement clock.
        """
        self.name: str = "docker"
        self._settings = settings
        self._cli = cli_path or settings.docker_cli_path
        self._clock: Clock = clock or SystemClock()
        self._sleeper: Sleeper = sleeper or AsyncSleeper()
        self._rng: RandomSource = rng or SeededRandom(seed=49357)
        self._breaker = CircuitBreaker(
            threshold=settings.circuit_failure_threshold,
            recovery_seconds=settings.circuit_recovery_timeout_s,
            clock=self._clock,
            name="docker-cli",
        )
        self._retry: RetryPolicy[ExecutionResult] = RetryPolicy(
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_s,
            max_delay=settings.retry_max_delay_s,
            sleeper=self._sleeper,
            rng=self._rng,
            breaker=self._breaker,
            retry_on=(asyncio.TimeoutError, OSError),
        )
        self._log = get_logger(__name__, component="backend", backend=self.name)

    @property
    def breaker(self) -> CircuitBreaker:
        """Expose the circuit breaker for health reporting.

        Returns:
            The shared :class:`CircuitBreaker` guarding docker CLI calls.
        """
        return self._breaker

    def _require_cli(self) -> str:
        """Resolve the docker binary or fail fast.

        Returns:
            Absolute path of the docker CLI.

        Raises:
            BackendUnavailableError: When docker is not installed.
        """
        resolved = shutil.which(self._cli)
        if resolved is None:
            raise BackendUnavailableError(
                f"docker CLI not found ({self._cli!r}); install docker or choose another backend",
            )
        return resolved

    async def health_check(self) -> bool:
        """Run ``docker version`` to confirm daemon availability.

        Returns:
            True when the daemon answers within 10 seconds.
        """
        try:
            cli = self._require_cli()
        except BackendUnavailableError:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                cli,
                "version",
                "--format",
                "{{.Server.Version}}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return await asyncio.wait_for(proc.wait(), timeout=10.0) == 0
        except (OSError, asyncio.TimeoutError):
            return False

    async def close(self) -> None:
        """No persistent handles; present for port conformance."""
        return None

    def build_run_command(self, spec: ExecutionSpec) -> list[str]:
        """Assemble the hardened ``docker run`` argv for one job.

        Args:
            spec: Resolved execution description.

        Returns:
            Full argv including interpreter and program entrypoint.

        Raises:
            BackendUnavailableError: If the docker CLI is missing.
        """
        cli = self._require_cli()
        container_name = f"{self._settings.docker_container_prefix}-{spec.execution_id.replace('_', '-')}"
        entry = f"/work/{spec.entrypoint_name}"
        from sandbox_service.models import Language

        inner: Sequence[str]
        if spec.language is Language.PYTHON:
            inner = ["python3", entry, *spec.args]
        else:
            inner = ["bash", entry, *spec.args]

        cpus = max(0.1, min(8.0, spec.limits.cpu_seconds / max(1.0, spec.limits.timeout_seconds)))
        env_flags: list[str] = ["-e", "PYTHONDONTWRITEBYTECODE=1"]
        for key, value in spec.env.items():
            env_flags += ["-e", f"{key}={value}"]

        argv: list[str] = [cli, "run", "--rm", "--name", container_name]
        argv += [
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m,noexec",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--memory", f"{spec.limits.memory_mb}m",
            "--memory-swap", f"{spec.limits.memory_mb}m",  # equal to --memory disables swap
            "--cpus", f"{cpus:.2f}",
            "--pids-limit", str(spec.limits.max_processes),
            "-v", f"{spec.cwd}:/work:rw",
            "-w", "/work",
            *env_flags,
            self._settings.docker_image,
            *inner,
        ]
        return argv

    async def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Execute one job in a fresh container with bounded output.

        Args:
            spec: Resolved description; ``spec.cwd`` is mounted at ``/work``.

        Returns:
            Typed result; docker-level failures map to ``status=error`` after
            retries are exhausted.

        Raises:
            BackendUnavailableError: If the docker CLI is missing entirely.
            CircuitOpenError: If the breaker rejects the call.
        """
        argv = self.build_run_command(spec)
        self._log.debug("docker_run", argv=shlex.join(argv), execution_id=spec.execution_id)

        async def _attempt() -> ExecutionResult:
            started = self._clock.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            budget = spec.limits.max_output_bytes
            per_stream = max(64, budget // 2)
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(input=(spec.stdin or "").encode("utf-8")),
                    timeout=spec.limits.timeout_seconds + 5.0,  # +daemon overhead grace
                )
            except asyncio.TimeoutError as exc:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                raise exc
            duration_ms = int((self._clock.monotonic() - started) * 1000)
            truncated = len(out) >= per_stream or len(err) >= per_stream
            exit_code = proc.returncode
            timed_out = exit_code == 137  # SIGKILL from our watchdog or OOM killer
            if timed_out:
                status = ExecutionStatus.TIMED_OUT
            elif exit_code == 0:
                status = ExecutionStatus.SUCCEEDED
            else:
                status = ExecutionStatus.FAILED
            return ExecutionResult(
                execution_id=spec.execution_id,
                status=status,
                exit_code=exit_code,
                stdout=out[:per_stream].decode("utf-8", errors="replace"),
                stderr=err[:per_stream].decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
                truncated=truncated,
                timed_out=timed_out,
                backend=self.name,
            )

        try:
            return await self._retry.execute(_attempt, description=f"docker run {spec.execution_id}")
        except asyncio.TimeoutError as exc:
            self._breaker.record_failure()
            return ExecutionResult(
                execution_id=spec.execution_id,
                status=ExecutionStatus.TIMED_OUT,
                timed_out=True,
                stderr=f"container exceeded wall-clock limit of {spec.limits.timeout_seconds:g}s",
                backend=self.name,
            )
        except (OSError,) as exc:
            self._breaker.record_failure()
            return ExecutionResult(
                execution_id=spec.execution_id,
                status=ExecutionStatus.ERROR,
                error="docker_invocation_failed",
                stderr=str(exc),
                backend=self.name,
            )
