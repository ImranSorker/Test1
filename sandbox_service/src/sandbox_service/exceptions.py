"""Custom exception hierarchy for the sandboxed code execution service.

Every error raised by this package derives from :class:`SandboxError`, so callers
can catch a single base class. Each exception carries a machine-readable
``code`` string that the HTTP API maps into structured error responses.

Example:
    >>> from sandbox_service.exceptions import ExecutionTimeoutError
    >>> try:
    ...     raise ExecutionTimeoutError("too slow")
    ... except SandboxError as exc:
    ...     assert exc.code == "execution_timeout"
"""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for all sandbox-service errors.

    Attributes:
        message: Human-readable error description.
        code: Stable machine-readable error identifier (snake_case).
        details: Optional structured context about the failure.
    """

    code: str = "sandbox_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of what went wrong.
            details: Optional mapping with structured context (logged and returned
                by the HTTP adapter in error payloads).
        """
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, object] = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        """Serialize the error for structured logging / JSON responses.

        Returns:
            A dict with ``error``, ``message`` and ``details`` keys.

        Example:
            >>> SandboxError("boom").to_dict()
            {'error': 'sandbox_error', 'message': 'boom', 'details': {}}
        """
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        """Return the human-readable message."""
        return self.message


class ConfigurationError(SandboxError):
    """Raised when settings are invalid or an environment is misconfigured.

    Example:
        >>> raise ConfigurationError("SANDBOX_API_PORT must be 1-65535")
    """

    code = "configuration_error"


class BackendUnavailableError(SandboxError):
    """Raised when the selected execution backend cannot be reached or started.

    Typical causes: missing ``docker`` binary, daemon not running, remote service
    down after retries were exhausted.
    """

    code = "backend_unavailable"


class CircuitOpenError(SandboxError):
    """Raised when the circuit breaker is open and calls are being rejected.

    Attributes:
        retry_after_seconds: Estimated seconds until the breaker allows a probe.
    """

    code = "circuit_open"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize with the breaker recovery window.

        Args:
            message: Human-readable description.
            retry_after_seconds: Seconds until the breaker transitions to half-open.
            details: Optional structured context.
        """
        super().__init__(message, details=details)
        self.retry_after_seconds: float = retry_after_seconds
        self.details["retry_after_seconds"] = retry_after_seconds


class ServiceBusyError(SandboxError):
    """Raised when the concurrency limit is reached and no slot is available.

    Back-pressure signal for agent orchestration: callers should retry after
    ``retry_after_seconds`` (surfaced as an HTTP ``Retry-After`` header).

    Attributes:
        retry_after_seconds: Conservative estimate of when a slot frees up.

    Example:
        >>> err = ServiceBusyError("queue full", retry_after_seconds=2.5)
        >>> err.to_dict()["details"]["retry_after_seconds"]
        2.5
    """

    code = "service_busy"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize with the estimated wait window.

        Args:
            message: Human-readable description.
            retry_after_seconds: Seconds until a slot is likely free.
            details: Optional structured context.
        """
        super().__init__(message, details=details)
        self.retry_after_seconds: float = retry_after_seconds
        self.details["retry_after_seconds"] = retry_after_seconds


class LanguageNotSupportedError(SandboxError):
    """Raised when a request asks for a language outside the configured allowlist.

    Example:
        >>> raise LanguageNotSupportedError("ruby", allowed=["python", "bash"])
    """

    code = "language_not_supported"


class UnsafeCodeError(SandboxError):
    """Raised when static analysis rejects a submission before execution.

    Attributes:
        findings: List of policy violations detected by the safety checker.
    """

    code = "unsafe_code"

    def __init__(self, message: str, findings: list[str], *, details: dict[str, object] | None = None) -> None:
        """Initialize with the list of policy findings.

        Args:
            message: Human-readable summary.
            findings: Individual policy violation descriptions.
            details: Optional extra structured context.
        """
        merged: dict[str, object] = dict(details or {})
        merged["findings"] = list(findings)
        super().__init__(message, details=merged)
        self.findings: list[str] = list(findings)


class ValidationError_(SandboxError):
    """Raised when client input fails domain-model validation.

    Named with a trailing underscore to avoid shadowing pydantic's
    ``ValidationError`` inside modules that import both.
    """

    code = "invalid_request"


class ExecutionTimeoutError(SandboxError):
    """Raised internally when a sandboxed process exceeds its wall-clock limit.

    The service normally converts this into an ``ExecutionResult`` with
    ``status=timed_out`` rather than surfacing it to callers; adapters raise it
    to signal that the kernel killed the job.
    """

    code = "execution_timeout"


class WorkspaceError(SandboxError):
    """Raised when workspace creation, file injection, or cleanup fails."""

    code = "workspace_error"


class OutputLimitExceededError(SandboxError):
    """Raised when captured output exceeds the configured byte budget.

    Normally the service truncates instead of raising; adapters may raise this
    when strict mode is requested.
    """

    code = "output_limit_exceeded"
