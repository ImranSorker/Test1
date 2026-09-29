"""Testing utilities: deterministic in-memory backends and fakes.

Import-safe for production code too — :class:`FakeBackend` is a legitimate
value of ``SANDBOX_DEFAULT_BACKEND=fake`` used by smoke tests, CI pipelines,
and agent-environment harnesses that must not spawn processes.

Example:
    >>> import asyncio
    >>> from sandbox_service.models import ExecutionResult, ExecutionStatus
    >>> from sandbox_service.testing import FakeBackend
    >>> fake = FakeBackend(records=[ExecutionResult(
    ...     execution_id="auto", status=ExecutionStatus.SUCCEEDED, stdout="hi\\n")])
    >>> async def demo() -> str:
    ...     from sandbox_service.interfaces import ExecutionSpec
    ...     result = await fake.run(ExecutionSpec(
    ...         execution_id="exe_1", language="python", command=[], source="print('hi')"))
    ...     return result.stdout
    >>> asyncio.run(demo())
    'hi\n'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from sandbox_service.exceptions import BackendUnavailableError
from sandbox_service.interfaces import Clock, ExecutionSpec
from sandbox_service.models import ExecutionResult, ExecutionStatus
from sandbox_service.observability import SystemClock, get_logger


class FakeBackend:
    """In-memory :class:`~sandbox_service.interfaces.ExecutionBackend` for tests.

    Behaviour knobs (all constructor-injected; no globals, no monkeypatching):

    * ``records``: scripted results popped one per :meth:`run`; the last
      record repeats forever once the list is exhausted.
    * ``handler``: optional ``spec -> ExecutionResult`` callable, consulted
      before ``records`` — use it to assert on specs or echo inputs.
    * ``raise_on_run``: exception instance raised by every :meth:`run`
      (e.g. ``BackendUnavailableError("boom")``) to exercise failure paths.
    * ``healthy``: fixed answer returned by :meth:`health_check`.

    Attributes:
        name: Always ``"fake"``.
        calls: Every :class:`ExecutionSpec` received, in order — the primary
            assertion surface for tests.
    """

    def __init__(
        self,
        *,
        records: list[ExecutionResult] | None = None,
        handler: Callable[[ExecutionSpec], ExecutionResult] | None = None,
        raise_on_run: BaseException | None = None,
        healthy: bool = True,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the fake backend.

        Args:
            records: Scripted results consumed in order (last one repeats).
            handler: Synchronous spec-to-result mapping with precedence over
                ``records``.
            raise_on_run: If set, :meth:`run` raises this exception after
                recording the spec (failure-path injection).
            healthy: Value returned by :meth:`health_check`.
            clock: Injectable clock stamped onto synthesized results.
        """
        self.name: str = "fake"
        self.calls: list[ExecutionSpec] = []
        self._records: list[ExecutionResult] = list(records or [])
        self._handler = handler
        self._raise_on_run = raise_on_run
        self._healthy = healthy
        self._clock: Clock = clock or SystemClock()
        self._closed = False
        self._log = get_logger(__name__, component="backend", backend=self.name)

    def _next_record(self) -> ExecutionResult | None:
        """Pop the next scripted record, keeping the final one sticky.

        Returns:
            The next :class:`ExecutionResult`, or ``None`` when no records
            were supplied.
        """
        if not self._records:
            return None
        if len(self._records) == 1:
            return self._records[0]
        return self._records.pop(0)

    async def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Record ``spec`` and produce the scripted/handler/default result.

        Args:
            spec: Resolved execution description.

        Returns:
            A result tagged ``backend="fake"`` with the real execution id and
            creation timestamp applied. When nothing is scripted, echoes a
            successful result whose stdout is the spec's source (handy for
            round-trip assertions).

        Raises:
            BackendUnavailableError: If ``raise_on_run`` was configured with
                that type; any other configured exception is raised verbatim.
            RuntimeError: If the backend has been closed (use-after-close is
                a bug worth surfacing loudly).
        """
        if self._closed:
            raise RuntimeError("FakeBackend used after close()")
        self.calls.append(spec)
        if self._raise_on_run is not None:
            raise self._raise_on_run
        if self._handler is not None:
            base = self._handler(spec)
        else:
            record = self._next_record()
            base = record if record is not None else ExecutionResult(
                execution_id="",
                status=ExecutionStatus.SUCCEEDED,
                exit_code=0,
                stdout=spec.source,
            )
        return base.model_copy(
            update={
                "execution_id": spec.execution_id,
                "backend": self.name,
                "created_at": self._clock.now(),
            }
        )

    async def health_check(self) -> bool:
        """Return the configured fixed health value.

        Returns:
            The ``healthy`` constructor argument.
        """
        return self._healthy

    async def close(self) -> None:
        """Mark the backend closed. Idempotent."""
        if not self._closed:
            self._closed = True
            self._log.info("fake_backend_closed", recorded_calls=len(self.calls))

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            ``self``.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Exit the async context manager, closing the backend.

        Args:
            exc_type: In-flight exception type, if any.
            exc: In-flight exception, if any.
            tb: Traceback object, if any.
        """
        await self.close()


def unavailable_error(message: str = "fake backend unavailable") -> BackendUnavailableError:
    """Build a canonical :class:`BackendUnavailableError` for failure injection.

    Args:
        message: Human-readable cause.

    Returns:
        The exception instance, ready for ``FakeBackend(raise_on_run=...)``.

    Example:
        >>> unavailable_error("boom").code
        'backend_unavailable'
    """
    return BackendUnavailableError(message)
