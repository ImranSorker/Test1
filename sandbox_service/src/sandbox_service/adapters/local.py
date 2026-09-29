"""Local process backend: run snippets as resource-limited child processes.

Kernel strategy (POSIX): the child is launched through a tiny bootstrap that
applies :func:`resource.setrlimit` (CPU time, address space, file size, process
count) *before* exec'ing the interpreter, and into its own process group so the
parent can ``killpg`` the whole tree on wall-clock timeout. On Windows only the
asyncio wall-clock timeout applies — health checks report this honestly.

This adapter satisfies :class:`~sandbox_service.interfaces.ExecutionBackend`.

Example:
    >>> from sandbox_service.config import Settings
    >>> from sandbox_service.models import ExecutionResult
    >>> backend = LocalProcessBackend(Settings())
    >>> backend.name
    'local'
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

from sandbox_service.config import Settings
from sandbox_service.exceptions import BackendUnavailableError
from sandbox_service.interfaces import Clock, ExecutionSpec
from sandbox_service.models import ExecutionResult, ExecutionStatus, Language
from sandbox_service.observability import SystemClock, get_logger

# Bootstrap executed by a clean `python -I -S -c` child. It sets rlimits and
# then re-execs the real interpreter. Both placeholders are filled with
# repr()s of plain ints/strings computed by the service — user code never
# contributes to this template's structure.
_BOOTSTRAP_TEMPLATE = """
import os
limits = {limits!r}
try:
    import resource
    if limits['cpu']:
        resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu'], limits['cpu'] + 1))
    if limits['as']:
        resource.setrlimit(resource.RLIMIT_AS, (limits['as'], limits['as']))
    if limits['fsize']:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits['fsize'], limits['fsize']))
    if limits['nproc']:
        resource.setrlimit(resource.RLIMIT_NPROC, (limits['nproc'], limits['nproc']))
except (ImportError, ValueError, OSError):
    pass  # platform without the resource module; wall-clock timeout still applies
{execv_call}
"""


def _entrypoint_filename(language: Language) -> str:
    """Return the workspace filename used for a language's program text.

    Args:
        language: Submission language.

    Returns:
        A safe relative filename such as ``"__main__.py"`` or ``"main.sh"``.
    """
    return "__main__.py" if language is Language.PYTHON else "main.sh"


class _CappedStreamReader:
    """Drains a child's stdout/stderr concurrently with a combined byte cap.

    Reads never stop (surplus data is discarded) so the child cannot block on a
    full pipe, while :meth:`collect` returns at most ``budget`` bytes total.

    Attributes:
        budget: Combined byte cap across both streams.
    """

    def __init__(
        self,
        stdout: asyncio.StreamReader | None,
        stderr: asyncio.StreamReader | None,
        budget: int,
    ) -> None:
        """Initialize the reader.

        Args:
            stdout: Child stdout stream or ``None``.
            stderr: Child stderr stream or ``None``.
            budget: Positive combined byte cap.
        """
        self.budget: int = max(128, budget)
        self._stdout = stdout
        self._stderr = stderr
        self._out = bytearray()
        self._err = bytearray()
        self._per_stream = max(64, self.budget // 2)
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        """Launch the two drain tasks. Call exactly once."""
        if self._stdout is not None:
            self._tasks.append(asyncio.ensure_future(self._drain(self._stdout, self._out)))
        if self._stderr is not None:
            self._tasks.append(asyncio.ensure_future(self._drain(self._stderr, self._err)))

    async def _drain(self, stream: asyncio.StreamReader, sink: bytearray) -> None:
        """Read ``stream`` to EOF, keeping only the first slice of ``sink``.

        Args:
            stream: Pipe reader.
            sink: Bytearray receiving capped output.
        """
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            room = self._per_stream - len(sink)
            if room > 0:
                sink.extend(chunk[:room])

    def completed(self) -> asyncio.Future[list[object]]:
        """Return a future resolving when both drains finish.

        Returns:
            Awaitable gathering the drain tasks (or an already-done future
            when there were no streams).
        """
        if not self._tasks:
            done: asyncio.Future[list[object]] = asyncio.get_event_loop().create_future()
            done.set_result([])
            return done
        return asyncio.gather(*self._tasks)

    async def cancel(self) -> None:
        """Cancel pending drain tasks (used after killing the child)."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, OSError, RuntimeError):
                await task

    def collect(self) -> tuple[bytes, bytes]:
        """Return captured bytes so far.

        Returns:
            ``(stdout, stderr)`` each capped at half the combined budget.
        """
        return bytes(self._out), bytes(self._err)


class LocalProcessBackend:
    """Execution kernel using local subprocesses guarded by rlimits + timeouts.

    Attributes:
        name: Always ``"local"``.
    """

    def __init__(self, settings: Settings, clock: Clock | None = None) -> None:
        """Initialize the backend.

        Args:
            settings: Resolved interpreter paths and limit ceilings.
            clock: Injectable clock for duration measurement.

        Raises:
            BackendUnavailableError: If configured interpreters cannot be found.
        """
        self.name: str = "local"
        self._settings = settings
        self._clock: Clock = clock or SystemClock()
        self._log = get_logger(__name__, component="backend", backend=self.name)
        try:
            self._python = settings.resolved_python_binary()
            self._bash = settings.resolved_bash_binary() if "bash" in settings.allowed_languages else ""
        except ValueError as exc:
            raise BackendUnavailableError(f"local interpreter missing: {exc}") from exc

    async def health_check(self) -> bool:
        """Verify interpreters exist and are executable.

        Returns:
            True when every allowed interpreter resolves on disk.
        """
        python_ok = Path(self._python).exists()
        bash_ok = (not self._bash) or Path(self._bash).exists()
        return python_ok and bash_ok

    async def close(self) -> None:
        """No persistent handles to release; present for port conformance."""
        return None

    def _interpreter_for(self, language: Language) -> str:
        """Map a language to its interpreter binary path.

        Args:
            language: Requested language.

        Returns:
            Absolute interpreter path.

        Raises:
            BackendUnavailableError: If the interpreter was not resolved at
                construction time (e.g. bash disabled but requested later).
        """
        if language is Language.PYTHON:
            return self._python
        if language is Language.BASH:
            if not self._bash:
                raise BackendUnavailableError("bash interpreter unavailable")
            return self._bash
        raise BackendUnavailableError(f"no interpreter for language {language!r}")

    def _build_launch(self, spec: ExecutionSpec) -> tuple[list[str], dict[str, str]]:
        """Build argv/environment that applies rlimits before running the job.

        Args:
            spec: Resolved execution description.

        Returns:
            ``(argv, env)`` ready for :meth:`asyncio.create_subprocess_exec`.

        Raises:
            BackendUnavailableError: If the language has no interpreter.
        """
        interpreter = self._interpreter_for(spec.language)
        entry = spec.cwd / _entrypoint_filename(spec.language)
        program_argv = [interpreter, str(entry), *spec.args]

        env = {"PATH": os.defpath + ":/usr/bin:/bin", "HOME": str(spec.cwd), "LANG": "C.UTF-8"}
        env.update(spec.env)

        if not self._settings.supports_rlimits:
            return program_argv, env

        limits = {
            "cpu": int(spec.limits.cpu_seconds),
            "as": int(spec.limits.memory_mb) * 1024 * 1024,
            "fsize": int(spec.limits.max_file_size_mb) * 1024 * 1024,
            "nproc": int(spec.limits.max_processes),
        }
        bootstrap = _BOOTSTRAP_TEMPLATE.format(
            limits=limits,
            execv_call=f"os.execv({interpreter!r}, {program_argv!r})",
        )
        # -I isolates the bootstrap from user site-packages and env vars;
        # -S skips site initialization for speed. The re-exec'd interpreter
        # starts fresh with the sanitized environment we pass via env=.
        launch = [sys.executable, "-I", "-S", "-c", bootstrap]
        return launch, env

    async def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Execute one job locally and capture bounded output.

        Args:
            spec: Resolved execution description; ``spec.cwd`` must already
                contain the entrypoint file written by the service layer.

        Returns:
            Typed result with truncated stdout/stderr and terminal status.

        Raises:
            BackendUnavailableError: If the interpreter cannot be launched.
        """
        argv, env = self._build_launch(spec)
        budget = spec.limits.max_output_bytes
        started = self._clock.monotonic()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(spec.cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group -> killpg works
            )
        except (OSError, ValueError) as exc:
            self._log.error("spawn_failed", execution_id=spec.execution_id, error=str(exc))
            raise BackendUnavailableError(f"failed to spawn sandbox process: {exc}") from exc

        payload = (spec.stdin or "").encode("utf-8")
        reader = _CappedStreamReader(proc.stdout, proc.stderr, budget)
        reader.start()
        stdin_task = self._feed_stdin(proc, payload)

        exit_code: int | None
        try:
            await asyncio.wait_for(reader.completed(), timeout=spec.limits.timeout_seconds)
            exit_code = await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, TimeoutError):
            timed_out = True
            self._kill_tree(proc)
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError, OSError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            await reader.cancel()
            exit_code = None
        finally:
            if stdin_task is not None and not stdin_task.done():
                stdin_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, OSError, RuntimeError):
                    await stdin_task

        duration_ms = int((self._clock.monotonic() - started) * 1000)
        stdout_bytes, stderr_bytes = reader.collect()
        truncated = min(len(stdout_bytes) + len(stderr_bytes), budget) >= budget
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if timed_out:
            status = ExecutionStatus.TIMED_OUT
            stderr = stderr or (
                f"killed: exceeded wall-clock limit of {spec.limits.timeout_seconds:g}s"
            )
        elif exit_code == 0:
            status = ExecutionStatus.SUCCEEDED
        else:
            status = ExecutionStatus.FAILED

        self._log.info(
            "execution_finished",
            execution_id=spec.execution_id,
            status=status.value,
            exit_code=exit_code,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
        return ExecutionResult(
            execution_id=spec.execution_id,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            truncated=truncated,
            timed_out=timed_out,
            backend=self.name,
        )

    @staticmethod
    def _feed_stdin(proc: asyncio.subprocess.Process, payload: bytes) -> asyncio.Task[None] | None:
        """Write stdin bytes and close the pipe without blocking the caller.

        Args:
            proc: Child process with an open stdin pipe.
            payload: Bytes to write (may be empty).

        Returns:
            The background write task, or ``None`` when stdin is unavailable.
        """
        if proc.stdin is None:
            return None

        async def _write_and_close() -> None:
            try:
                proc.stdin.write(payload)  # type: ignore[union-attr]
                await proc.stdin.drain()  # type: ignore[union-attr]
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if proc.stdin is not None:  # type: ignore[unreachable]
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        proc.stdin.close()

        return asyncio.ensure_future(_write_and_close())

    @staticmethod
    def _kill_tree(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the child's entire process group, falling back to the child.

        Args:
            proc: The timed-out sandbox process.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                proc.kill()
