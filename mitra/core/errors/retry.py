"""
🤖 Mitra AI - Retry Pattern
Automatic retry with exponential backoff for transient failures.
Coded by Denvil with love 🤍
"""

import asyncio
import functools
import random
from typing import Callable, Any, Optional, Tuple, Type, TypeVar, Union
from dataclasses import dataclass

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add randomization to delays
    retry_on: Tuple[Type[Exception], ...] = (Exception,)  # Exceptions to retry on
    exclude: Tuple[Type[Exception], ...] = ()  # Exceptions to never retry on


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay before next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))

    # Cap at maximum delay
    delay = min(delay, config.max_delay)

    # Add jitter (0.5x to 1.5x)
    if config.jitter:
        delay = delay * (0.5 + random.random())

    return delay


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    exclude: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Multiplier for exponential backoff
        jitter: Whether to add randomization to delays
        retry_on: Tuple of exception types to retry on
        exclude: Tuple of exception types to never retry on
        on_retry: Callback called before each retry (attempt, exception)

    Returns:
        Decorated function

    Example:
        @with_retry(max_attempts=3, base_delay=1.0)
        async def call_api():
            ...

        @with_retry(retry_on=(ConnectionError, TimeoutError))
        def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        exclude=exclude,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if exception is excluded
                    if isinstance(e, config.exclude):
                        logger.warning(
                            "retry_excluded_exception",
                            function=func.__name__,
                            attempt=attempt,
                            exception=type(e).__name__,
                        )
                        raise

                    # Check if exception is retryable
                    if not isinstance(e, config.retry_on):
                        raise

                    last_exception = e

                    # Don't retry if max attempts reached
                    if attempt >= config.max_attempts:
                        logger.error(
                            "retry_max_attempts_reached",
                            function=func.__name__,
                            attempts=attempt,
                            exception=str(e),
                        )
                        raise

                    # Calculate delay
                    delay = calculate_delay(attempt, config)

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay=delay,
                        exception=str(e),
                    )

                    # Call retry callback
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception as callback_error:
                            logger.warning(
                                "retry_callback_error",
                                error=str(callback_error),
                            )

                    # Wait before retry
                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if exception is excluded
                    if isinstance(e, config.exclude):
                        logger.warning(
                            "retry_excluded_exception",
                            function=func.__name__,
                            attempt=attempt,
                            exception=type(e).__name__,
                        )
                        raise

                    # Check if exception is retryable
                    if not isinstance(e, config.retry_on):
                        raise

                    last_exception = e

                    # Don't retry if max attempts reached
                    if attempt >= config.max_attempts:
                        logger.error(
                            "retry_max_attempts_reached",
                            function=func.__name__,
                            attempts=attempt,
                            exception=str(e),
                        )
                        raise

                    # Calculate delay
                    delay = calculate_delay(attempt, config)

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay=delay,
                        exception=str(e),
                    )

                    # Call retry callback
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception as callback_error:
                            logger.warning(
                                "retry_callback_error",
                                error=str(callback_error),
                            )

                    # Wait before retry
                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations.

    Usage:
        async with RetryableOperation(max_attempts=3) as retry:
            for attempt in retry:
                try:
                    result = await do_something()
                    break
                except Exception as e:
                    await attempt.handle_exception(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )
        self._attempt = 0
        self._last_exception: Optional[Exception] = None

    async def __aenter__(self) -> "RetryableOperation":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    def __iter__(self) -> "RetryableOperation":
        self._attempt = 0
        return self

    def __next__(self) -> "RetryAttempt":
        self._attempt += 1
        if self._attempt > self.config.max_attempts:
            if self._last_exception:
                raise self._last_exception
            raise StopIteration

        return RetryAttempt(self, self._attempt)


@dataclass
class RetryAttempt:
    """Represents a single retry attempt."""
    operation: RetryableOperation
    attempt_number: int

    async def handle_exception(self, exception: Exception) -> None:
        """Handle an exception during this attempt."""
        self.operation._last_exception = exception

        if self.attempt_number >= self.operation.config.max_attempts:
            raise exception

        delay = calculate_delay(self.attempt_number, self.operation.config)
        logger.warning(
            "retryable_operation_attempt",
            attempt=self.attempt_number,
            max_attempts=self.operation.config.max_attempts,
            delay=delay,
            exception=str(exception),
        )
        await asyncio.sleep(delay)
