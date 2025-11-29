"""
🤖 Mitra AI - Custom Exception Classes
Specialized exceptions for different error scenarios.
Coded by Denvil with love 🤍
"""

from typing import Optional, Any
from .base import MitraError, ErrorCategory, ErrorSeverity, ErrorContext


class AIEngineError(MitraError):
    """Error in the AI engine processing."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.AI_ENGINE,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
        )
        self.model_name = model_name


class TelegramError(MitraError):
    """Error in Telegram bot operations."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        chat_id: Optional[int] = None,
        update_id: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TELEGRAM,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
        )
        self.chat_id = chat_id
        self.update_id = update_id


class ValidationError(MitraError):
    """Error in input validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            context=context,
        )
        self.field = field
        self.value = value


class DatabaseError(MitraError):
    """Error in database operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
            recoverable=True,
        )
        self.operation = operation
        self.table = table


class AuthenticationError(MitraError):
    """Error in authentication."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.WARNING,
            context=context,
            user_facing_message="Authentication failed. Please try again.",
        )
        if context:
            context.user_id = user_id


class RateLimitError(MitraError):
    """Error when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        retry_after: Optional[float] = None,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            context=context,
            user_facing_message=f"Rate limit exceeded. Please wait {int(retry_after or 60)} seconds.",
            recoverable=True,
        )
        self.user_id = user_id
        self.retry_after = retry_after


class ConfigurationError(MitraError):
    """Error in configuration."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recoverable=False,
        )
        self.config_key = config_key


class ModelLoadError(MitraError):
    """Error loading AI model."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            original_error=original_error,
            recoverable=False,
        )
        self.model_name = model_name


class InferenceError(MitraError):
    """Error during model inference."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_tokens: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
        )
        self.model_name = model_name
        self.input_tokens = input_tokens


class NetworkError(MitraError):
    """Error in network operations."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
        )
        self.url = url
        self.status_code = status_code


class ExternalServiceError(MitraError):
    """Error from external service."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.ERROR,
            context=context,
            original_error=original_error,
        )
        self.service_name = service_name
