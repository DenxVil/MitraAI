"""
🤖 Mitra AI - Error Handling System
Comprehensive error handling with circuit breaker and retry patterns.
Coded by Denvil with love 🤍
"""

from .base import ErrorSeverity, ErrorCategory, ErrorContext, MitraError
from .exceptions import (
    AIEngineError,
    TelegramError,
    ValidationError,
    DatabaseError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError,
    ModelLoadError,
    InferenceError,
)
from .handler import ErrorHandler, GlobalErrorHandler, handle_errors
from .circuit_breaker import CircuitBreaker, CircuitState
from .retry import with_retry, RetryConfig

__all__ = [
    # Base
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "MitraError",
    # Exceptions
    "AIEngineError",
    "TelegramError",
    "ValidationError",
    "DatabaseError",
    "AuthenticationError",
    "RateLimitError",
    "ConfigurationError",
    "ModelLoadError",
    "InferenceError",
    # Handler
    "ErrorHandler",
    "GlobalErrorHandler",
    "handle_errors",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    # Retry
    "with_retry",
    "RetryConfig",
]
