"""
🤖 Mitra AI - Logging Middleware
Handles request logging for the bot.
Coded by Denvil with love 🤍
"""

from typing import Optional
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import TelegramConfig


class LoggingMiddleware:
    """
    Logging middleware for Telegram bot.

    Features:
    - Request logging
    - User action tracking
    - Performance metrics
    - Error tracking
    """

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config

    async def process(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """
        Log the incoming update.

        Returns:
            Always True (logging doesn't block processing).
        """
        # Skip if logging is disabled
        if not self.config.log_messages and not self.config.log_user_actions:
            return True

        user = update.effective_user
        chat = update.effective_chat

        # Build log data
        log_data = {
            "update_id": update.update_id,
            "user_id": user.id if user else None,
            "username": user.username if user else None,
            "chat_id": chat.id if chat else None,
            "chat_type": chat.type if chat else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Determine update type and log appropriately
        if update.message:
            await self._log_message(update.message, log_data)
        elif update.callback_query:
            await self._log_callback(update.callback_query, log_data)
        elif update.inline_query:
            await self._log_inline_query(update.inline_query, log_data)
        elif update.edited_message:
            await self._log_edited_message(update.edited_message, log_data)

        # Store request timestamp for performance tracking
        context.user_data["request_start"] = datetime.now(timezone.utc)

        return True

    async def _log_message(self, message, log_data: dict) -> None:
        """Log a message update."""
        msg_type = self._determine_message_type(message)
        log_data["update_type"] = "message"
        log_data["message_type"] = msg_type
        log_data["message_id"] = message.message_id

        if self.config.log_messages and message.text:
            # Truncate long messages
            text = message.text[:100] + "..." if len(message.text) > 100 else message.text
            log_data["message_preview"] = text

        logger.info("telegram_message_received", **log_data)

    async def _log_callback(self, callback_query, log_data: dict) -> None:
        """Log a callback query."""
        log_data["update_type"] = "callback_query"
        log_data["callback_data"] = callback_query.data

        logger.info("telegram_callback_received", **log_data)

    async def _log_inline_query(self, inline_query, log_data: dict) -> None:
        """Log an inline query."""
        log_data["update_type"] = "inline_query"
        log_data["query"] = inline_query.query[:50] if inline_query.query else None

        logger.info("telegram_inline_query_received", **log_data)

    async def _log_edited_message(self, message, log_data: dict) -> None:
        """Log an edited message."""
        log_data["update_type"] = "edited_message"
        log_data["message_id"] = message.message_id

        logger.info("telegram_message_edited", **log_data)

    def _determine_message_type(self, message) -> str:
        """Determine the type of message."""
        if message.text:
            if message.text.startswith("/"):
                return "command"
            return "text"
        elif message.photo:
            return "photo"
        elif message.voice:
            return "voice"
        elif message.document:
            return "document"
        elif message.video:
            return "video"
        elif message.audio:
            return "audio"
        elif message.sticker:
            return "sticker"
        elif message.location:
            return "location"
        elif message.contact:
            return "contact"
        else:
            return "other"

    async def log_response(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        response_type: str = "message",
    ) -> None:
        """Log a bot response (call this after processing)."""
        request_start = context.user_data.get("request_start")
        if request_start:
            duration = (datetime.now(timezone.utc) - request_start).total_seconds()
            logger.info(
                "telegram_response_sent",
                update_id=update.update_id,
                response_type=response_type,
                duration_seconds=duration,
            )
