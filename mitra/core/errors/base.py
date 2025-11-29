"""
🤖 Mitra AI - Base Error Classes
Foundation for the error handling system.
Coded by Denvil with love 🤍
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()


class ErrorCategory(str, Enum):
    """Categories of errors in the system."""
    AI_ENGINE = "ai_engine"
    TELEGRAM = "telegram"
    VALIDATION = "validation"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    MODEL = "model"
    INFERENCE = "inference"
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class ErrorContext:
    """Context information for an error."""
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.name,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "request_id": self.request_id,
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "additional_data": self.additional_data,
        }


class MitraError(Exception):
    """
    Base exception for all Mitra AI errors.
    Provides structured error information with context.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        user_facing_message: Optional[str] = None,
        recoverable: bool = True,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(category=category, severity=severity)
        self.original_error = original_error
        self.user_facing_message = user_facing_message or self._default_user_message()
        self.recoverable = recoverable
        self.error_code = error_code or self._generate_error_code()
        self.timestamp = datetime.now(timezone.utc)

    def _default_user_message(self) -> str:
        """Generate a default user-facing error message."""
        messages = {
            ErrorCategory.AI_ENGINE: "I'm having trouble processing your request right now. Please try again in a moment.",
            ErrorCategory.TELEGRAM: "There was an issue with the messaging service. Please try again.",
            ErrorCategory.RATE_LIMIT: "You're sending messages too quickly. Please wait a moment before trying again.",
            ErrorCategory.VALIDATION: "I couldn't understand that request. Could you please rephrase?",
            ErrorCategory.DATABASE: "There was an issue accessing data. Please try again.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.CONFIGURATION: "There's a configuration issue. Please contact support.",
            ErrorCategory.MODEL: "The AI model is temporarily unavailable. Please try again later.",
            ErrorCategory.INFERENCE: "I had trouble generating a response. Please try again.",
            ErrorCategory.NETWORK: "Network connection issue. Please check your connection and try again.",
            ErrorCategory.INTERNAL: "Something went wrong on my end. Please try again.",
            ErrorCategory.EXTERNAL: "An external service is unavailable. Please try again later.",
        }
        return messages.get(self.category, "An unexpected error occurred. Please try again.")

    def _generate_error_code(self) -> str:
        """Generate a unique error code."""
        import hashlib
        content = f"{self.category.value}:{self.message}:{self.timestamp.isoformat()}"
        return f"MITRA-{hashlib.sha256(content.encode()).hexdigest()[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.name,
            "user_facing_message": self.user_facing_message,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "original_error": str(self.original_error) if self.original_error else None,
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.category.value}: {self.message}"

    def __repr__(self) -> str:
        return f"MitraError(code={self.error_code}, category={self.category.value}, message={self.message!r})"
