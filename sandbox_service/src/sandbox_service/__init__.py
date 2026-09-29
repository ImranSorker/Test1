"""Sandboxed code execution service — run untrusted snippets, safely.

A local-first, provider-agnostic kernel for agent workflows: submit Python or
Bash source, get a typed :class:`~sandbox_service.models.ExecutionResult` with
captured output, exit status, timing, and static-safety findings.

Three interchangeable surfaces share one application service:

* **Python API** — :class:`~sandbox_service.service.SandboxService`
* **HTTP API** — ``uvicorn sandbox_service.api:create_app --factory``
* **CLI** — ``sandbox run|serve|health`` (see :mod:`sandbox_service.cli`)

Example:
    >>> import asyncio
    >>> from sandbox_service import SandboxService, ExecutionRequest
    >>> async def demo() -> str:
    ...     async with SandboxService() as svc:
    ...         result = await svc.execute(ExecutionRequest(language="python", source="print(6*7)"))
    ...         return result.stdout.strip()
    >>> asyncio.run(demo())
    '42'
"""

from __future__ import annotations

from sandbox_service.config import Settings
from sandbox_service.exceptions import (
    BackendUnavailableError,
    CircuitOpenError,
    ConfigurationError,
    ExecutionTimeoutError,
    LanguageNotSupportedError,
    OutputLimitExceededError,
    SandboxError,
    UnsafeCodeError,
    ValidationError_,
    WorkspaceError,
)
from sandbox_service.interfaces import (
    Clock,
    ExecutionBackend,
    ExecutionSpec,
    RandomSource,
    Sleeper,
    WorkspaceManager,
)
from sandbox_service.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    HealthReport,
    InputFile,
    Language,
    ResourceLimits,
    SafetyFinding,
)
from sandbox_service.safety import SafetyChecker, SafetyReport
from sandbox_service.service import SandboxService, build_backend

__version__ = "0.1.0"

__all__ = [
    "BackendUnavailableError",
    "CircuitOpenError",
    "Clock",
    "ConfigurationError",
    "ExecutionBackend",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionSpec",
    "ExecutionStatus",
    "ExecutionTimeoutError",
    "HealthReport",
    "InputFile",
    "Language",
    "LanguageNotSupportedError",
    "OutputLimitExceededError",
    "RandomSource",
    "ResourceLimits",
    "SafetyChecker",
    "SafetyFinding",
    "SafetyReport",
    "SandboxError",
    "SandboxService",
    "Settings",
    "Sleeper",
    "UnsafeCodeError",
    "ValidationError_",
    "WorkspaceError",
    "WorkspaceManager",
    "__version__",
    "build_backend",
]
