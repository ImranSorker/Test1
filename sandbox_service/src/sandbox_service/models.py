"""Domain models for the sandboxed code execution service.

This is the innermost layer: it imports nothing from adapters or services and
depends only on pydantic and the standard library. Everything crossing a public
boundary (CLI, HTTP, Python API) is represented by one of these models.

Example:
    >>> req = ExecutionRequest(language="python", source="print(1+1)")
    >>> (req.limits or ResourceLimits()).timeout_seconds
    10.0
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime  # noqa: TC003  (used at runtime by pydantic)
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


class Language(str, Enum):
    """Execution languages supported by the service."""

    PYTHON = "python"
    BASH = "bash"


class ExecutionStatus(str, Enum):
    """Terminal state of an execution job."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    ERROR = "error"


class ResourceLimits(BaseModel):
    """Resource budget applied to a single execution.

    All values are validated against hard bounds supplied via
    :class:`~sandbox_service.config.Settings` at request-normalization time;
    the model itself enforces absolute safety floors/ceilings so that no
    instance can ever encode "unlimited".

    Attributes:
        timeout_seconds: Wall-clock limit; the kernel kills the job past this.
        memory_mb: Address-space cap per process in mebibytes.
        max_output_bytes: Budget for captured stdout/stderr combined.
        cpu_seconds: CPU-time cap (rlimit ``RLIMIT_CPU``) as a backstop.
        max_file_size_mb: Per-file write cap inside the workspace.
        max_processes: Process/thread cap (``RLIMIT_NPROC``).
    """

    model_config = ConfigDict(frozen=True)

    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 10.0
    memory_mb: Annotated[int, Field(gt=0, le=8192)] = 512
    max_output_bytes: Annotated[int, Field(ge=128, le=4 * 1024 * 1024)] = 65536
    cpu_seconds: Annotated[int, Field(gt=0, le=600)] = 10
    max_file_size_mb: Annotated[int, Field(gt=0, le=512)] = 16
    max_processes: Annotated[int, Field(ge=1, le=1024)] = 64

    @model_validator(mode="after")
    def _cpu_not_exceeding_wallclock_grace(self) -> "ResourceLimits":
        """Keep CPU seconds sane relative to wall clock (allow small rounding).

        Raises:
            ValueError: If ``cpu_seconds`` is more than ``timeout_seconds + 5``.
        """
        if self.cpu_seconds > self.timeout_seconds + 5:
            raise ValueError(
                f"cpu_seconds ({self.cpu_seconds}) must not exceed timeout_seconds + 5 "
                f"({self.timeout_seconds + 5:g})"
            )
        return self


class InputFile(BaseModel):
    """A text file injected into the sandbox workspace before execution.

    Attributes:
        path: Workspace-relative path (e.g. ``data/input.txt``). Absolute paths
            and parent-directory traversal are rejected.
        content: UTF-8 text contents.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    content: str = ""

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        """Reject unsafe workspace paths.

        Args:
            value: Candidate relative path.

        Returns:
            The validated path, unchanged.

        Raises:
            ValueError: If the path is absolute, contains ``..``, starts with
                ``/``, or embeds null bytes.
        """
        if not value:
            raise ValueError("file path must not be empty")
        if "\x00" in value:
            raise ValueError("file path must not contain null bytes")
        if value.startswith("/") or value.startswith("\\"):
            raise ValueError(f"file path must be workspace-relative, got {value!r}")
        parts = value.replace("\\", "/").split("/")
        if any(part == ".." for part in parts):
            raise ValueError(f"file path must not traverse upward, got {value!r}")
        return value


class ExecutionRequest(BaseModel):
    """A client submission asking the service to run code.

    Example:
        >>> ExecutionRequest(
        ...     language="python",
        ...     source="import pathlib; print(pathlib.Path('in.txt').read_text())",
        ...     files=[InputFile(path="in.txt", content="hello")],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    language: Language
    source: Annotated[str, Field(min_length=1, max_length=200_000)]
    args: list[str] = Field(default_factory=list)
    stdin: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    files: list[InputFile] = Field(default_factory=list)
    limits: ResourceLimits | None = None
    # When True the static-safety checker rejects the submission instead of
    # merely annotating the result.
    enforce_safety: bool = False

    @field_validator("args")
    @classmethod
    def _args_are_strings(cls, value: list[str]) -> list[str]:
        """Validate CLI arguments passed to the program.

        Args:
            value: Argument list.

        Returns:
            The validated argument list.

        Raises:
            ValueError: If any argument contains null bytes.
        """
        for arg in value:
            if "\x00" in arg:
                raise ValueError("arguments must not contain null bytes")
        return value

    @field_validator("env")
    @classmethod
    def _validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate environment variable names injected into the sandbox.

        Args:
            value: Mapping of name to value.

        Returns:
            The validated mapping.

        Raises:
            ValueError: On malformed identifiers or null bytes in values.
        """
        for name, val in value.items():
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            if "\x00" in val:
                raise ValueError(f"environment value for {name!r} must not contain null bytes")
        return value


class SafetyFinding(BaseModel):
    """One policy violation detected by the static safety checker.

    Attributes:
        rule: Stable rule identifier (e.g. ``dangerous-call:os.system``).
        message: Human-readable explanation.
        line: 1-based source line number, when known.
        severity: ``"block"`` findings fail the request under
            ``enforce_safety``; ``"warn"`` findings only annotate the result.
    """

    model_config = ConfigDict(frozen=True)

    rule: str
    message: str
    line: int | None = None
    severity: Annotated[str, Field(pattern="^(block|warn)$")] = "warn"


class ExecutionResult(BaseModel):
    """Outcome of one sandboxed execution — the typed "Result" of the service.

    Attributes:
        execution_id: Server-assigned unique id for the job.
        status: Terminal :class:`ExecutionStatus`.
        exit_code: Process exit code, or ``None`` if the process never started
            or was killed by the kernel.
        stdout: Captured standard output (truncated to the byte budget).
        stderr: Captured standard error (truncated to the byte budget).
        duration_ms: Measured wall-clock duration in milliseconds.
        truncated: True if stdout/stderr were cut to fit ``max_output_bytes``.
        timed_out: True if the job was killed for exceeding its time limit.
        findings: Static-safety findings collected before execution.
        backend: Name of the adapter that ran the job.
        error: Machine-readable error code when ``status`` is ``error``.
        created_at: UTC timestamp assigned by the injected clock.
    """

    model_config = ConfigDict(frozen=True)

    execution_id: str
    status: ExecutionStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    truncated: bool = False
    timed_out: bool = False
    findings: list[SafetyFinding] = Field(default_factory=list)
    backend: str = "unknown"
    error: str | None = None
    created_at: datetime | None = None

    @property
    def ok(self) -> bool:
        """Whether the job finished successfully.

        Returns:
            True iff ``status`` is :attr:`ExecutionStatus.SUCCEEDED`.
        """
        return self.status is ExecutionStatus.SUCCEEDED


class HealthReport(BaseModel):
    """Backend health snapshot returned by ``GET /healthz`` and the CLI.

    Attributes:
        healthy: Whether the active backend can accept work right now.
        backend: Active backend name.
        circuit_state: Circuit-breaker state (``closed``/``open``/``half_open``).
        details: Adapter-specific information (image name, interpreter paths...).
    """

    model_config = ConfigDict(frozen=True)

    healthy: bool
    backend: str
    circuit_state: str = "closed"
    details: dict[str, Any] = Field(default_factory=dict)


def new_execution_id() -> str:
    """Generate a unique, URL-safe execution identifier.

    Returns:
        A string like ``"exe_7c9e6679-..."`` suitable for logs and REST paths.

    Example:
        >>> assert new_execution_id().startswith("exe_")
    """
    return f"exe_{uuid.uuid4()}"
