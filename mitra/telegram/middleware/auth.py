"""
🤖 Mitra AI - Authentication Middleware
Handles user authentication and authorization.
Coded by Denvil with love 🤍
"""

from typing import Optional, Set
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


class AuthMiddleware:
    """
    Authentication middleware for Telegram bot.

    Features:
    - User authentication
    - Admin verification
    - Maintenance mode handling
    - User blocking
    """

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self._blocked_users: Set[int] = set()
        self._admin_users: Set[int] = config.admin_ids

    async def process(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """
        Process authentication for an update.

        Returns:
            True if update should continue processing, False to stop.
        """
        user = update.effective_user
        if not user:
            # No user, might be a channel post or anonymous update
            return True

        # Check if user is blocked
        if await self.is_blocked(user.id):
            logger.warning("blocked_user_attempt", user_id=user.id)
            return False

        # Check maintenance mode
        if self.config.maintenance_mode:
            if not await self.is_admin(user.id):
                await self._send_maintenance_message(update, context)
                return False

        # Store user info in context
        context.user_data["user_id"] = user.id
        context.user_data["username"] = user.username
        context.user_data["is_admin"] = await self.is_admin(user.id)
        context.user_data["last_active"] = datetime.now(timezone.utc).isoformat()

        return True

    async def is_admin(self, user_id: int) -> bool:
        """Check if a user is an admin."""
        return user_id in self._admin_users

    async def is_blocked(self, user_id: int) -> bool:
        """Check if a user is blocked."""
        return user_id in self._blocked_users

    async def block_user(self, user_id: int) -> None:
        """Block a user."""
        self._blocked_users.add(user_id)
        logger.info("user_blocked", user_id=user_id)

    async def unblock_user(self, user_id: int) -> None:
        """Unblock a user."""
        self._blocked_users.discard(user_id)
        logger.info("user_unblocked", user_id=user_id)

    async def add_admin(self, user_id: int) -> None:
        """Add a user as admin."""
        self._admin_users.add(user_id)
        logger.info("admin_added", user_id=user_id)

    async def remove_admin(self, user_id: int) -> None:
        """Remove admin privileges from a user."""
        self._admin_users.discard(user_id)
        logger.info("admin_removed", user_id=user_id)

    async def _send_maintenance_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send maintenance mode message to user."""
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    "🔧 *Maintenance Mode*\n\n"
                    "The bot is currently under maintenance. "
                    "Please try again later.\n\n"
                    "_Coded by Denvil with love 🤍_"
                ),
                parse_mode="Markdown",
            )
