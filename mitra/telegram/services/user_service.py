"""
🤖 Mitra AI - User Service
Handles user data and preferences.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, field

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class TelegramUser:
    """Telegram user data."""
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    language_code: Optional[str] = None
    is_premium: bool = False
    is_admin: bool = False
    is_blocked: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)


class UserService:
    """
    Service for managing Telegram users.

    Features:
    - User CRUD operations
    - Settings management
    - Statistics tracking
    """

    def __init__(self) -> None:
        # In-memory storage (replace with database in production)
        self._users: Dict[int, TelegramUser] = {}

    async def get_user(self, user_id: int) -> Optional[TelegramUser]:
        """Get a user by ID."""
        return self._users.get(user_id)

    async def get_or_create_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        language_code: Optional[str] = None,
    ) -> TelegramUser:
        """Get an existing user or create a new one."""
        if user_id in self._users:
            user = self._users[user_id]
            # Update user info
            user.username = username or user.username
            user.first_name = first_name or user.first_name
            user.last_name = last_name or user.last_name
            user.language_code = language_code or user.language_code
            user.last_active = datetime.now(timezone.utc)
            return user

        # Create new user
        user = TelegramUser(
            user_id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            language_code=language_code,
        )
        self._users[user_id] = user

        logger.info(
            "user_created",
            user_id=user_id,
            username=username,
        )

        return user

    async def update_user(
        self,
        user_id: int,
        **kwargs: Any,
    ) -> Optional[TelegramUser]:
        """Update user properties."""
        user = await self.get_user(user_id)
        if not user:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.last_active = datetime.now(timezone.utc)
        return user

    async def increment_message_count(self, user_id: int) -> int:
        """Increment user's message count."""
        user = await self.get_user(user_id)
        if user:
            user.message_count += 1
            return user.message_count
        return 0

    async def get_setting(
        self,
        user_id: int,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a user setting."""
        user = await self.get_user(user_id)
        if user:
            return user.settings.get(key, default)
        return default

    async def set_setting(
        self,
        user_id: int,
        key: str,
        value: Any,
    ) -> bool:
        """Set a user setting."""
        user = await self.get_user(user_id)
        if user:
            user.settings[key] = value
            return True
        return False

    async def delete_setting(
        self,
        user_id: int,
        key: str,
    ) -> bool:
        """Delete a user setting."""
        user = await self.get_user(user_id)
        if user and key in user.settings:
            del user.settings[key]
            return True
        return False

    async def block_user(self, user_id: int) -> bool:
        """Block a user."""
        user = await self.get_user(user_id)
        if user:
            user.is_blocked = True
            logger.info("user_blocked", user_id=user_id)
            return True
        return False

    async def unblock_user(self, user_id: int) -> bool:
        """Unblock a user."""
        user = await self.get_user(user_id)
        if user:
            user.is_blocked = False
            logger.info("user_unblocked", user_id=user_id)
            return True
        return False

    async def set_admin(self, user_id: int, is_admin: bool = True) -> bool:
        """Set admin status for a user."""
        user = await self.get_user(user_id)
        if user:
            user.is_admin = is_admin
            logger.info("admin_status_changed", user_id=user_id, is_admin=is_admin)
            return True
        return False

    async def get_all_users(self) -> List[TelegramUser]:
        """Get all users."""
        return list(self._users.values())

    async def get_active_users(self, days: int = 7) -> List[TelegramUser]:
        """Get users active in the last N days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            user for user in self._users.values()
            if user.last_active >= cutoff
        ]

    async def get_user_count(self) -> int:
        """Get total user count."""
        return len(self._users)

    async def get_admin_users(self) -> List[TelegramUser]:
        """Get all admin users."""
        return [
            user for user in self._users.values()
            if user.is_admin
        ]

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            logger.info("user_deleted", user_id=user_id)
            return True
        return False
