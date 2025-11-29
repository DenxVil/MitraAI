"""
🤖 Mitra AI - Circuit Breaker Pattern
Prevents cascading failures by stopping requests to failing services.
Coded by Denvil with love 🤍
"""

import asyncio
import functools
from enum import Enum, auto
from typing import Callable, Any, Optional, TypeVar
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = auto()  # Normal operation, requests pass through
    OPEN = auto()  # Failure threshold reached, requests blocked
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: float = 30.0  # Seconds before attempting recovery
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0
    times_opened: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered

    Usage:
        circuit = CircuitBreaker("external_api")

        @circuit
        async def call_external_api():
            ...
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        excluded_exceptions: tuple = (),
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            excluded_exceptions=excluded_exceptions,
        )
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def _check_state(self) -> None:
        """Check and update circuit state based on timeout."""
        if self._state == CircuitState.OPEN and self._opened_at:
            timeout_delta = timedelta(seconds=self.config.timeout)
            if datetime.now(timezone.utc) - self._opened_at >= timeout_delta:
                logger.info(
                    "circuit_breaker_half_open",
                    name=self.name,
                    timeout=self.config.timeout,
                )
                self._state = CircuitState.HALF_OPEN
                self._stats.success_count = 0

    async def _on_success(self) -> None:
        """Handle successful request."""
        self._stats.success_count += 1
        self._stats.total_successes += 1
        self._stats.last_success_time = datetime.now(timezone.utc)

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.success_count >= self.config.success_threshold:
                logger.info(
                    "circuit_breaker_closed",
                    name=self.name,
                    success_count=self._stats.success_count,
                )
                self._state = CircuitState.CLOSED
                self._stats.failure_count = 0
                self._opened_at = None

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed request."""
        # Check if exception is excluded
        if isinstance(error, self.config.excluded_exceptions):
            return

        self._stats.failure_count += 1
        self._stats.total_failures += 1
        self._stats.last_failure_time = datetime.now(timezone.utc)

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open state re-opens the circuit
            logger.warning(
                "circuit_breaker_reopened",
                name=self.name,
                error=str(error),
            )
            self._state = CircuitState.OPEN
            self._opened_at = datetime.now(timezone.utc)
            self._stats.times_opened += 1
        elif self._state == CircuitState.CLOSED:
            if self._stats.failure_count >= self.config.failure_threshold:
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_count=self._stats.failure_count,
                    error=str(error),
                )
                self._state = CircuitState.OPEN
                self._opened_at = datetime.now(timezone.utc)
                self._stats.times_opened += 1

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            await self._check_state()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name,
                    timeout_remaining=self._get_timeout_remaining(),
                )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                await self._on_success()

            return result

        except Exception as e:
            async with self._lock:
                await self._on_failure(e)
            raise

    def _get_timeout_remaining(self) -> Optional[float]:
        """Get remaining timeout before recovery attempt."""
        if self._opened_at:
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            remaining = self.config.timeout - elapsed
            return max(0, remaining)
        return None

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at = None
        logger.info("circuit_breaker_reset", name=self.name)

    def __call__(self, func: F) -> F:
        """Decorator form of circuit breaker."""

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""

    def __init__(
        self,
        message: str,
        circuit_name: str,
        timeout_remaining: Optional[float] = None,
    ):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.timeout_remaining = timeout_remaining
