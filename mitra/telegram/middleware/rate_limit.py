"""
🤖 Mitra AI - Rate Limit Middleware
Handles rate limiting for bot requests.
Coded by Denvil with love 🤍
"""

from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from telegram import Update
from telegram.ext import ContextTypes

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import TelegramConfig


@dataclass
class UserRateLimit:
    """Tracks rate limit for a user."""
    user_id: int
    message_count: int = 0
    command_count: int = 0
    media_count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def reset_if_expired(self, window_seconds: int) -> None:
        """Reset counts if window has expired."""
        now = datetime.now(timezone.utc)
        if (now - self.window_start).total_seconds() >= window_seconds:
            self.message_count = 0
            self.command_count = 0
            self.media_count = 0
            self.window_start = now


class RateLimitMiddleware:
    """
    Rate limiting middleware for Telegram bot.

    Features:
    - Per-user rate limiting
    - Separate limits for messages, commands, and media
    - Sliding window implementation
    - Admin bypass option
    """

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self._user_limits: Dict[int, UserRateLimit] = {}

    async def process(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """
        Process rate limiting for an update.

        Returns:
            True if update should continue processing, False if rate limited.
        """
        user = update.effective_user
        if not user:
            return True

        # Admins bypass rate limiting
        if context.user_data.get("is_admin", False):
            return True

        # Get or create user rate limit
        user_limit = self._get_user_limit(user.id)

        # Check window expiration
        user_limit.reset_if_expired(self.config.rate_limit_window_seconds)

        # Determine update type and check limit
        if update.message:
            if update.message.text and update.message.text.startswith("/"):
                # Command
                if not await self._check_command_limit(user_limit, update, context):
                    return False
            elif update.message.photo or update.message.voice or update.message.document:
                # Media
                if not await self._check_media_limit(user_limit, update, context):
                    return False
            else:
                # Regular message
                if not await self._check_message_limit(user_limit, update, context):
                    return False

        return True

    def _get_user_limit(self, user_id: int) -> UserRateLimit:
        """Get or create a user rate limit tracker."""
        if user_id not in self._user_limits:
            self._user_limits[user_id] = UserRateLimit(user_id=user_id)
        return self._user_limits[user_id]

    async def _check_message_limit(
        self,
        user_limit: UserRateLimit,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Check message rate limit."""
        if user_limit.message_count >= self.config.rate_limit_messages_per_minute:
            await self._send_rate_limit_message(update, context, "messages")
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_limit.user_id,
                type="messages",
                count=user_limit.message_count,
            )
            return False

        user_limit.message_count += 1
        return True

    async def _check_command_limit(
        self,
        user_limit: UserRateLimit,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Check command rate limit."""
        if user_limit.command_count >= self.config.rate_limit_commands_per_minute:
            await self._send_rate_limit_message(update, context, "commands")
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_limit.user_id,
                type="commands",
                count=user_limit.command_count,
            )
            return False

        user_limit.command_count += 1
        return True

    async def _check_media_limit(
        self,
        user_limit: UserRateLimit,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Check media rate limit."""
        if user_limit.media_count >= self.config.rate_limit_media_per_minute:
            await self._send_rate_limit_message(update, context, "media")
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_limit.user_id,
                type="media",
                count=user_limit.media_count,
            )
            return False

        user_limit.media_count += 1
        return True

    async def _send_rate_limit_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        limit_type: str,
    ) -> None:
        """Send rate limit exceeded message to user."""
        if update.effective_chat:
            remaining = self.config.rate_limit_window_seconds
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    f"⏳ *Rate Limit Exceeded*\n\n"
                    f"You're sending too many {limit_type}. "
                    f"Please wait {remaining} seconds before trying again.\n\n"
                    "_Coded by Denvil with love 🤍_"
                ),
                parse_mode="Markdown",
            )

    def get_remaining(self, user_id: int, limit_type: str = "messages") -> int:
        """Get remaining requests for a user."""
        if user_id not in self._user_limits:
            if limit_type == "messages":
                return self.config.rate_limit_messages_per_minute
            elif limit_type == "commands":
                return self.config.rate_limit_commands_per_minute
            else:
                return self.config.rate_limit_media_per_minute

        user_limit = self._user_limits[user_id]
        user_limit.reset_if_expired(self.config.rate_limit_window_seconds)

        if limit_type == "messages":
            return max(0, self.config.rate_limit_messages_per_minute - user_limit.message_count)
        elif limit_type == "commands":
            return max(0, self.config.rate_limit_commands_per_minute - user_limit.command_count)
        else:
            return max(0, self.config.rate_limit_media_per_minute - user_limit.media_count)

    def reset_user(self, user_id: int) -> None:
        """Reset rate limit for a user."""
        if user_id in self._user_limits:
            del self._user_limits[user_id]
        logger.info("rate_limit_reset", user_id=user_id)
