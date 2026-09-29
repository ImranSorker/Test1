"""Port definitions (Protocol interfaces) for the sandbox execution service.

Every external dependency — the execution kernel, the clock, the random source,
the workspace filesystem — is expressed here as a :class:`typing.Protocol` so
concrete adapters are swappable without touching domain or service code.

Example:
    >>> class AlwaysOk:
    ...     name = "ok"
    ...     async def run(self, spec): raise AssertionError("unused")
    ...     async def health_check(self): return True
    ...     async def close(self): return None
    >>> isinstance(AlwaysOk(), ExecutionBackend)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sandbox_service.models import ExecutionResult, Language, ResourceLimits


@dataclass(frozen=True)
class ExecutionSpec:
    """Fully-resolved, validated description of one job handed to a backend.

    Built by the application layer from an ``ExecutionRequest`` + ``Settings``;
    backends must not re-interpret policy.

    Attributes:
        execution_id: Unique id assigned by the service.
        language: Resolved interpreter selection.
        command: Argv used to launch the program (interpreter + entrypoint).
        source: Original program text (for logs and remote injection).
        args: Command-line arguments appended after ``command``.
        stdin: Text fed to the process' standard input.
        env: Environment variables for the process (already merged/sanitized).
        cwd: Host directory that the backend should expose as the workdir.
        entrypoint_name: Workspace-relative filename holding ``source``
            (written by the service before the backend runs).
        limits: Enforced resource budget.
        extra: Adapter-specific hints (e.g. docker image overrides).
    """

    execution_id: str
    language: Language
    command: list[str]
    source: str
    args: list[str] = field(default_factory=list)
    stdin: str = ""
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path = Path(".")
    entrypoint_name: str = "__main__.py"
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    """A kernel that can run one :class:`ExecutionSpec` in isolation.

    Implementations: :class:`~sandbox_service.adapters.local.LocalProcessBackend`,
    :class:`~sandbox_service.adapters.docker.DockerBackend`,
    :class:`~sandbox_service.adapters.remote.HttpRemoteBackend`,
    and ``FakeBackend`` for tests.
    """

    @property
    def name(self) -> str:
        """Short unique identifier for the backend (used in results/logs)."""
        ...

    async def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Execute one job and return its typed result.

        Args:
            spec: Fully-resolved execution description.

        Returns:
            An :class:`~sandbox_service.models.ExecutionResult` with captured
            output already truncated to the byte budget.

        Raises:
            BackendUnavailableError: If the kernel cannot accept work.
            ExecutionTimeoutError: Only if the backend killed the job and the
                adapter chooses to signal it instead of returning a result.
        """
        ...

    async def health_check(self) -> bool:
        """Probe whether the backend can currently execute jobs.

        Returns:
            True when healthy, False otherwise. Never raises for expected
            failure modes; unexpected ones propagate.
        """
        ...

    async def close(self) -> None:
        """Release sockets, clients, and child-process handles. Idempotent."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Time source abstraction (injectable for deterministic tests)."""

    def now(self) -> datetime:
        """Return the current UTC time.

        Returns:
            A timezone-aware ``datetime`` in UTC.
        """
        ...

    def monotonic(self) -> float:
        """Return a monotonically increasing seconds value for duration math.

        Returns:
            Seconds since an arbitrary fixed point.
        """
        ...


@runtime_checkable
class Sleeper(Protocol):
    """Async sleep abstraction so retries/backoff are testable without waiting."""

    async def sleep(self, seconds: float) -> None:
        """Suspend the coroutine.

        Args:
            seconds: Duration to sleep.
        """
        ...


@runtime_checkable
class RandomSource(Protocol):
    """Deterministic randomness port used for retry jitter."""

    def uniform(self, low: float, high: float) -> float:
        """Return a float in ``[low, high]``.

        Args:
            low: Lower bound.
            high: Upper bound.

        Returns:
            A pseudorandom float within the bounds.
        """
        ...


@runtime_checkable
class WorkspaceManager(Protocol):
    """Creates and disposes isolated working directories for jobs."""

    async def write_files(self, workspace: Path, files: list[Any]) -> list[Path]:
        """Inject client-supplied input files into a workspace asynchronously.

        Args:
            workspace: Directory previously returned by :meth:`create`.
            files: Validated :class:`~sandbox_service.models.InputFile` items.

        Returns:
            Absolute paths of the written files, in input order.

        Raises:
            WorkspaceError: On unsafe paths (traversal/symlink escape) or I/O
                failures.
        """
        ...

    def create(self, execution_id: str) -> Path:
        """Create a fresh workspace directory for one job.

        Args:
            execution_id: Unique job id used to name the directory.

        Returns:
            The absolute path of the new workspace.

        Raises:
            WorkspaceError: If the directory could not be created.
        """
        ...

    def cleanup(self, path: Path) -> None:
        """Remove a workspace directory tree.

        Args:
            path: Directory previously returned by :meth:`create`.

        Raises:
            WorkspaceError: If removal failed for an unexpected reason.
        """
        ...
