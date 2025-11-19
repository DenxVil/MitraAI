"""
Structured logging configuration for Mitra AI.

Provides centralized logging with correlation IDs, structured output,
and appropriate formatting for different environments.
"""

import logging
import sys
from typing import Any, Dict
import structlog
from structlog.types import FilteringBoundLogger


def setup_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: The environment (development, staging, production)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Determine if we should use pretty console output or JSON
    use_json = environment in ("staging", "production")

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if use_json:
        # Production: JSON output for log aggregation
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Pretty console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        A configured structured logger
    """
    return structlog.get_logger(name)


def add_correlation_id(correlation_id: str) -> None:
    """
    Add a correlation ID to the logging context.
    
    Args:
        correlation_id: The correlation/request ID to track
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation ID from the logging context."""
    structlog.contextvars.unbind_contextvars("correlation_id")


def log_event(
    logger: FilteringBoundLogger,
    event: str,
    level: str = "info",
    **kwargs: Any
) -> None:
    """
    Log a structured event with additional context.
    
    Args:
        logger: The logger instance
        event: The event description
        level: Log level (debug, info, warning, error, critical)
        **kwargs: Additional context to include in the log
    """
    log_method = getattr(logger, level.lower())
    log_method(event, **kwargs)
