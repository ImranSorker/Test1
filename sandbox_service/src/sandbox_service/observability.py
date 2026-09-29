"""Shared infrastructure: structured logging, clock/sleep/random adapters.

These are the default implementations for the injectable ports defined in
:mod:`sandbox_service.interfaces`. They live in one small module because they
are trivial glue around the standard library; swapping them is done by passing
different objects to :class:`~sandbox_service.service.SandboxService` — never
by monkeypatching globals.

Example:
    >>> import io
    >>> configure_logging(level="WARNING", stream=io.StringIO())
    >>> log = get_logger("demo", component="example")
    >>> log.info("ignored because below WARNING")
"""

from __future__ import annotations

import asyncio
import logging as _logging
import random
import sys
import time
from datetime import UTC, datetime

import structlog


def configure_logging(level: str = "INFO", *, stream: object | None = None) -> None:
    """Configure structlog to emit single-line JSON logs via ``logging``.

    Idempotent: safe to call multiple times (e.g. from CLI and tests).

    Args:
        level: One of ``DEBUG|INFO|WARNING|ERROR|CRITICAL`` (case-insensitive).
        stream: Output stream for log records; defaults to stderr. Injected in
            tests to capture output without touching global state.

    Raises:
        ValueError: If ``level`` is not a recognized logging level name.
    """
    normalized = level.upper()
    numeric = getattr(_logging, normalized, None)
    if not isinstance(numeric, int):
        raise ValueError(f"invalid log level: {level!r}")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(
            file=sys.stderr if stream is None else stream  # type: ignore[arg-type]
        ),
        cache_logger_on_first_use=False,
    )

    root = _logging.getLogger()
    root.setLevel(numeric)


def get_logger(name: str, **initial_values: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger with contextual key/values attached.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        initial_values: Keys bound to every message from this logger.

    Returns:
        A :class:`structlog.stdlib.BoundLogger`.

    Example:
        >>> get_logger("svc", backend="local").debug("created")
    """
    return structlog.get_logger(name).bind(**initial_values)


class SystemClock:
    """Real wall-clock implementation of the :class:`~sandbox_service.interfaces.Clock` port."""

    def now(self) -> datetime:
        """Return current UTC time.

        Returns:
            Timezone-aware UTC ``datetime``.
        """
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return monotonic seconds for duration measurement.

        Returns:
            Seconds since an arbitrary fixed origin (``time.monotonic``).
        """
        return time.monotonic()


class AsyncSleeper:
    """Default :class:`~sandbox_service.interfaces.Sleeper` wrapping ``asyncio.sleep``."""

    async def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``.

        Args:
            seconds: Non-negative duration.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"sleep duration must be >= 0, got {seconds}")
        await asyncio.sleep(seconds)


class SeededRandom:
    """Deterministic :class:`~sandbox_service.interfaces.RandomSource` implementation.

    Attributes:
        seed: The seed supplied at construction (``None`` = OS entropy).
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG.

        Args:
            seed: Optional integer seed; pass one for reproducible jitter.
        """
        self.seed: int | None = seed
        self._rng: random.Random = random.Random(seed)

    def uniform(self, low: float, high: float) -> float:
        """Return a float in ``[low, high]``.

        Args:
            low: Lower bound.
            high: Upper bound.

        Returns:
            Pseudorandom float; deterministic when a seed was provided.

        Raises:
            ValueError: If ``low > high``.
        """
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        return self._rng.uniform(low, high)
