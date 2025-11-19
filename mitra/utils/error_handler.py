"""
Error handling and exception management for Mitra AI.

Provides custom exceptions, error categorization, and centralized error handling.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import traceback
from .logger import get_logger


logger = get_logger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors in the system."""
    AI_SERVICE = "ai_service"
    TELEGRAM_API = "telegram_api"
    CONFIGURATION = "configuration"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    SAFETY = "safety"
    INTERNAL = "internal"
    EXTERNAL_API = "external_api"


@dataclass
class ErrorContext:
    """Context information for an error."""
    category: ErrorCategory
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class MitraError(Exception):
    """Base exception for Mitra AI errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        user_facing_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.context = context or ErrorContext(category=category)
        self.original_error = original_error
        self.user_facing_message = user_facing_message or self._default_user_message()

    def _default_user_message(self) -> str:
        """Generate a default user-facing error message."""
        messages = {
            ErrorCategory.AI_SERVICE: "I'm having trouble processing your request right now. Please try again in a moment.",
            ErrorCategory.TELEGRAM_API: "There was an issue with the messaging service. Please try again.",
            ErrorCategory.RATE_LIMIT: "You're sending messages too quickly. Please wait a moment before trying again.",
            ErrorCategory.VALIDATION: "I couldn't understand that request. Could you please rephrase?",
            ErrorCategory.SAFETY: "I'm unable to respond to that type of request for safety reasons.",
            ErrorCategory.INTERNAL: "Something went wrong on my end. Please try again.",
        }
        return messages.get(self.category, "An unexpected error occurred. Please try again.")


class ErrorHandler:
    """Centralized error handling and logging."""
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: Optional[ErrorContext] = None,
        user_id: Optional[str] = None
    ) -> MitraError:
        """
        Handle an error and convert it to a MitraError.
        
        Args:
            error: The original exception
            context: Additional error context
            user_id: The user ID if available
            
        Returns:
            A MitraError instance with appropriate handling
        """
        # If already a MitraError, just log and return
        if isinstance(error, MitraError):
            ErrorHandler._log_error(error)
            return error

        # Determine error category
        category = ErrorHandler._categorize_error(error)
        
        # Create context if not provided
        if context is None:
            context = ErrorContext(category=category, user_id=user_id)
        elif user_id and not context.user_id:
            context.user_id = user_id

        # Create MitraError
        mitra_error = MitraError(
            message=str(error),
            category=category,
            context=context,
            original_error=error
        )

        # Log the error
        ErrorHandler._log_error(mitra_error)

        return mitra_error

    @staticmethod
    def _categorize_error(error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and properties."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Check for specific error patterns
        if "rate" in error_msg and "limit" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "api" in error_msg or "openai" in error_msg or "azure" in error_msg:
            return ErrorCategory.AI_SERVICE
        elif "telegram" in error_msg:
            return ErrorCategory.TELEGRAM_API
        elif "validation" in error_msg or "invalid" in error_msg:
            return ErrorCategory.VALIDATION
        elif "config" in error_msg or error_type in ("KeyError", "ValueError"):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.INTERNAL

    @staticmethod
    def _log_error(error: MitraError) -> None:
        """Log an error with full context."""
        logger.error(
            "mitra_error_occurred",
            error_message=error.message,
            error_category=error.category.value,
            user_id=error.context.user_id,
            conversation_id=error.context.conversation_id,
            message_id=error.context.message_id,
            additional_data=error.context.additional_data,
            stack_trace=traceback.format_exc() if error.original_error else None,
        )

    @staticmethod
    def get_user_message(error: Exception) -> str:
        """Get a user-friendly error message."""
        if isinstance(error, MitraError):
            return error.user_facing_message
        return "An unexpected error occurred. Please try again."
