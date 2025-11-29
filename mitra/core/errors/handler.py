"""
🤖 Mitra AI - Error Handler
Centralized error handling with routing and user messages.
Coded by Denvil with love 🤍
"""

import traceback
import functools
import asyncio
from typing import Optional, Callable, Any, Dict, Type, TypeVar, Union
from datetime import datetime, timezone

from .base import MitraError, ErrorCategory, ErrorSeverity, ErrorContext

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorHandler:
    """
    Centralized error handling and routing.
    Routes errors to appropriate handlers based on category.
    """

    def __init__(self) -> None:
        self._handlers: Dict[ErrorCategory, Callable[[MitraError], None]] = {}
        self._fallback_handler: Optional[Callable[[MitraError], None]] = None

    def register_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[MitraError], None],
    ) -> None:
        """Register a handler for a specific error category."""
        self._handlers[category] = handler

    def set_fallback_handler(
        self,
        handler: Callable[[MitraError], None],
    ) -> None:
        """Set the fallback handler for unhandled categories."""
        self._fallback_handler = handler

    def handle(self, error: Exception, **context_kwargs: Any) -> MitraError:
        """
        Handle an error and convert it to a MitraError.

        Args:
            error: The original exception
            **context_kwargs: Additional context information

        Returns:
            A MitraError instance with appropriate handling
        """
        # Convert to MitraError if needed
        if isinstance(error, MitraError):
            mitra_error = error
        else:
            category = self._categorize_error(error)
            context = ErrorContext(
                category=category,
                user_id=context_kwargs.get("user_id"),
                conversation_id=context_kwargs.get("conversation_id"),
                message_id=context_kwargs.get("message_id"),
                request_id=context_kwargs.get("request_id"),
                component=context_kwargs.get("component"),
                operation=context_kwargs.get("operation"),
                additional_data=context_kwargs.get("additional_data"),
            )
            mitra_error = MitraError(
                message=str(error),
                category=category,
                context=context,
                original_error=error,
            )

        # Log the error
        self._log_error(mitra_error)

        # Route to appropriate handler
        handler = self._handlers.get(mitra_error.category)
        if handler:
            try:
                handler(mitra_error)
            except Exception as handler_error:
                logger.error(
                    "error_handler_failed",
                    handler_error=str(handler_error),
                    original_error=str(mitra_error),
                )
        elif self._fallback_handler:
            try:
                self._fallback_handler(mitra_error)
            except Exception as fallback_error:
                logger.error(
                    "fallback_handler_failed",
                    handler_error=str(fallback_error),
                    original_error=str(mitra_error),
                )

        return mitra_error

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and properties."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        if "rate" in error_msg and "limit" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "api" in error_msg or "openai" in error_msg or "azure" in error_msg:
            return ErrorCategory.AI_ENGINE
        elif "telegram" in error_msg:
            return ErrorCategory.TELEGRAM
        elif "validation" in error_msg or "invalid" in error_msg:
            return ErrorCategory.VALIDATION
        elif "config" in error_msg or error_type in ("KeyError", "ValueError"):
            return ErrorCategory.CONFIGURATION
        elif "database" in error_msg or "sql" in error_msg:
            return ErrorCategory.DATABASE
        elif "auth" in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif "model" in error_msg or "load" in error_msg:
            return ErrorCategory.MODEL
        elif "network" in error_msg or "connection" in error_msg:
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.INTERNAL

    def _log_error(self, error: MitraError) -> None:
        """Log an error with full context."""
        log_data = error.to_dict()
        log_data["stack_trace"] = (
            traceback.format_exc() if error.original_error else None
        )

        if error.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL):
            logger.critical("mitra_critical_error", **log_data)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error("mitra_error", **log_data)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning("mitra_warning", **log_data)
        else:
            logger.info("mitra_info", **log_data)

    @staticmethod
    def get_user_message(error: Exception) -> str:
        """Get a user-friendly error message."""
        if isinstance(error, MitraError):
            return error.user_facing_message
        return "An unexpected error occurred. Please try again."


class GlobalErrorHandler:
    """
    Global error handler singleton.
    Provides application-wide error handling.
    """

    _instance: Optional["GlobalErrorHandler"] = None
    _handler: ErrorHandler

    def __new__(cls) -> "GlobalErrorHandler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handler = ErrorHandler()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "GlobalErrorHandler":
        """Get the global error handler instance."""
        return cls()

    def handle(self, error: Exception, **context_kwargs: Any) -> MitraError:
        """Handle an error globally."""
        return self._handler.handle(error, **context_kwargs)

    def register_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[MitraError], None],
    ) -> None:
        """Register a handler for a specific error category."""
        self._handler.register_handler(category, handler)

    def set_fallback_handler(
        self,
        handler: Callable[[MitraError], None],
    ) -> None:
        """Set the fallback handler."""
        self._handler.set_fallback_handler(handler)


def handle_errors(
    *,
    category: Optional[ErrorCategory] = None,
    user_message: Optional[str] = None,
    reraise: bool = False,
    log_level: ErrorSeverity = ErrorSeverity.ERROR,
    on_error: Optional[Callable[[MitraError], Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for handling errors in functions.

    Args:
        category: Default error category for uncategorized errors
        user_message: Custom user-facing message
        reraise: Whether to re-raise the error after handling
        log_level: Logging level for the error
        on_error: Callback to execute on error

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                mitra_error = _handle_exception(
                    e, func.__name__, category, user_message, log_level
                )
                if on_error:
                    on_error(mitra_error)
                if reraise:
                    raise mitra_error from e
                return None

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                mitra_error = _handle_exception(
                    e, func.__name__, category, user_message, log_level
                )
                if on_error:
                    on_error(mitra_error)
                if reraise:
                    raise mitra_error from e
                return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def _handle_exception(
    error: Exception,
    func_name: str,
    category: Optional[ErrorCategory],
    user_message: Optional[str],
    log_level: ErrorSeverity,
) -> MitraError:
    """Helper to handle exceptions in the decorator."""
    handler = GlobalErrorHandler.get_instance()

    if isinstance(error, MitraError):
        if user_message:
            error.user_facing_message = user_message
        return handler.handle(error, operation=func_name)

    context = ErrorContext(
        category=category or ErrorCategory.INTERNAL,
        severity=log_level,
        operation=func_name,
    )

    mitra_error = MitraError(
        message=str(error),
        category=category or ErrorCategory.INTERNAL,
        severity=log_level,
        context=context,
        original_error=error,
        user_facing_message=user_message,
    )

    return handler.handle(mitra_error, operation=func_name)
