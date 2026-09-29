"""Retry with exponential backoff + jitter, and a circuit breaker.

Pure-Policy helpers used by remote-ish adapters (docker CLI, HTTP). All time
and randomness enters through injected ports (:class:`Sleeper`,
:class:`RandomSource`, :class:`Clock`) so tests run instantly and
deterministically.

Example:
    >>> from sandbox_service.observability import AsyncSleeper
    >>> breaker = CircuitBreaker(threshold=2, recovery_seconds=5.0)
    >>> breaker.state
    'closed'
    >>> policy = RetryPolicy(
    ...     attempts=3, base_delay=0.1, max_delay=2.0, sleeper=AsyncSleeper()
    )
    >>> policy.compute_delay(2)
    0.2
"""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from sandbox_service.exceptions import CircuitOpenError
from sandbox_service.interfaces import Clock, RandomSource, Sleeper
from sandbox_service.observability import SystemClock, get_logger

T = TypeVar("T")


class CircuitState(str, enum.Enum):
    """Circuit-breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Counts consecutive failures and stops calls to a failing dependency.

    Behaviour: after ``threshold`` consecutive failures the circuit opens;
    while open, calls are rejected with :class:`CircuitOpenError`. Once
    ``recovery_seconds`` have elapsed (per the injected clock) the circuit
    moves to half-open and lets one probe through — success closes it, failure
    re-opens it.

    Attributes:
        threshold: Consecutive failures required to trip open.
        recovery_seconds: Cool-down before a half-open probe is allowed.
    """

    def __init__(
        self,
        *,
        threshold: int,
        recovery_seconds: float,
        clock: Clock | None = None,
        name: str = "default",
    ) -> None:
        """Initialize the breaker.

        Args:
            threshold: Consecutive failures before opening (>= 1).
            recovery_seconds: Seconds to wait before probing again.
            clock: Injectable time source; defaults to monotonic system time.
            name: Label used in log lines and error details.

        Raises:
            ValueError: If ``threshold`` < 1 or ``recovery_seconds`` <= 0.
        """
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        if clock is None:
            clock = SystemClock()
        self.threshold: int = threshold
        self.recovery_seconds: float = recovery_seconds
        self.name: str = name
        self._clock: Clock = clock
        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: float | None = None
        self._log = get_logger(__name__, component="circuit_breaker", breaker=name)

    @property
    def state(self) -> str:
        """Current breaker state as a plain string.

        Returns:
            One of ``"closed" | "open" | "half_open"`` — evaluating any pending
            recovery window against the injected clock.
        """
        self._evaluate_transition()
        return self._state.value

    @property
    def failure_count(self) -> int:
        """Consecutive failures recorded so far.

        Returns:
            Non-negative counter value.
        """
        return self._consecutive_failures

    def _evaluate_transition(self) -> None:
        """Move OPEN -> HALF_OPEN when the recovery window has elapsed."""
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            elapsed = self._clock.monotonic() - self._opened_at
            if elapsed >= self.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                self._log.info("breaker_half_open", elapsed_seconds=round(elapsed, 3))

    def allow(self) -> None:
        """Gate a call: raise if the circuit is open.

        Raises:
            CircuitOpenError: When the breaker is fully open and the recovery
                window has not yet elapsed.
        """
        self._evaluate_transition()
        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None  # invariant: set whenever OPEN
            remaining = max(0.0, self.recovery_seconds - (self._clock.monotonic() - self._opened_at))
            raise CircuitOpenError(
                f"circuit '{self.name}' is open; retry in {remaining:.1f}s",
                retry_after_seconds=remaining,
            )

    def record_success(self) -> None:
        """Reset failure counts and close the circuit."""
        was = self._state
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        if was is not CircuitState.CLOSED:
            self._log.info("breaker_closed", previous_state=was.value)

    def record_failure(self) -> None:
        """Register one failure; open the circuit at the threshold."""
        self._consecutive_failures += 1
        if self._state is CircuitState.HALF_OPEN or self._consecutive_failures >= self.threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock.monotonic()
            self._log.warning(
                "breaker_opened",
                consecutive_failures=self._consecutive_failures,
                threshold=self.threshold,
            )

    def reset(self) -> None:
        """Return the breaker to a pristine closed state (for tests/admin)."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None


class RetryPolicy(Generic[T]):
    """Executes an async callable with exponential backoff + full jitter.

    Example:
        >>> class InstantSleeper:
        ...     total = 0.0
        ...     async def sleep(self, s: float) -> None:
        ...         InstantSleeper.total += s
        >>> policy = RetryPolicy(
        ...     attempts=3, base_delay=0.1, max_delay=2.0,
        ...     sleeper=InstantSleeper(), rng=None, breaker=None,
        ... )
    """

    def __init__(
        self,
        *,
        attempts: int,
        base_delay: float,
        max_delay: float,
        sleeper: Sleeper,
        rng: RandomSource | None = None,
        breaker: CircuitBreaker | None = None,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        """Configure the policy.

        Args:
            attempts: Maximum number of invocations (>= 1).
            base_delay: First backoff step in seconds.
            max_delay: Ceiling for any single backoff step.
            sleeper: Injected sleep port (use a fake in tests).
            rng: Injected randomness for jitter; ``None`` disables jitter.
            breaker: Optional shared circuit breaker guarding the dependency.
            retry_on: Exception types considered transient/retryable.

        Raises:
            ValueError: On non-positive attempts or delays.
        """
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        if base_delay <= 0 or max_delay <= 0:
            raise ValueError("delays must be positive")
        self.attempts: int = attempts
        self.base_delay: float = base_delay
        self.max_delay: float = max_delay
        self.retry_on: tuple[type[BaseException], ...] = retry_on
        self._sleeper = sleeper
        self._rng = rng
        self._breaker = breaker
        self._log = get_logger(__name__, component="retry")

    @property
    def breaker(self) -> CircuitBreaker | None:
        """The guarded circuit breaker, if any.

        Returns:
            The breaker instance or ``None``.
        """
        return self._breaker

    def compute_delay(self, attempt: int) -> float:
        """Compute the backoff delay applied *after* failed attempt N.

        Exponential: ``base * 2**(attempt-1)``, capped at ``max_delay``, then
        scaled by full jitter ``U(0.5, 1.0)`` when an RNG is configured.

        Args:
            attempt: 1-based index of the attempt that just failed.

        Returns:
            Delay in seconds.

        Raises:
            ValueError: If ``attempt`` < 1.
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        raw = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        if self._rng is None:
            return raw
        return raw * self._rng.uniform(0.5, 1.0)

    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        description: str = "operation",
    ) -> T:
        """Run ``func`` under the retry/backoff/circuit policy.

        Args:
            func: Zero-argument coroutine factory invoked on each attempt.
            description: Label for logs and the final error message.

        Returns:
            Whatever ``func`` resolves to on the first successful attempt.

        Raises:
            CircuitOpenError: If the breaker rejects all attempts.
            Exception: The last retryable exception if attempts are exhausted
                (the original exception type is preserved and re-raised).
        """
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            if self._breaker is not None:
                self._breaker.allow()
            try:
                result = await func()
            except self.retry_on as exc:
                last_error = exc
                if self._breaker is not None:
                    self._breaker.record_failure()
                if attempt >= self.attempts:
                    break
                delay = self.compute_delay(attempt)
                self._log.warning(
                    "attempt_failed",
                    operation=description,
                    attempt=attempt,
                    attempts=self.attempts,
                    retry_in_seconds=round(delay, 3),
                    error_type=type(exc).__name__,
                )
                await self._sleeper.sleep(delay)
            else:
                if self._breaker is not None:
                    self._breaker.record_success()
                return result

        assert last_error is not None  # loop only exits via return or break-with-error
        self._log.error(
            "retries_exhausted",
            operation=description,
            attempts=self.attempts,
            error_type=type(last_error).__name__,
        )
        raise last_error
