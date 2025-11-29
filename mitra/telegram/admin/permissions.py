"""
🤖 Mitra AI - Admin Permissions
Role-Based Access Control for admin panel.
Coded by Denvil with love 🤍
"""

from enum import Enum, auto
from typing import Optional, Set, Dict, Callable, Any
from functools import wraps
from dataclasses import dataclass, field

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AdminRole(Enum):
    """Admin role levels."""
    SUPER_ADMIN = auto()  # Full access
    ADMIN = auto()  # Almost full access
    OPERATOR = auto()  # Can manage users and view stats
    MODERATOR = auto()  # Can moderate content
    VIEWER = auto()  # Read-only access


class Permission(Enum):
    """Individual permissions."""
    # User management
    VIEW_USERS = auto()
    MANAGE_USERS = auto()
    BAN_USERS = auto()
    UNBAN_USERS = auto()
    DELETE_USERS = auto()

    # Admin management
    VIEW_ADMINS = auto()
    MANAGE_ADMINS = auto()
    ASSIGN_ROLES = auto()

    # AI control
    VIEW_AI_STATUS = auto()
    CONFIGURE_AI = auto()
    RETRAIN_AI = auto()
    BENCHMARK_AI = auto()

    # Monitoring
    VIEW_DASHBOARD = auto()
    VIEW_LOGS = auto()
    VIEW_METRICS = auto()
    VIEW_ERRORS = auto()

    # Broadcast
    SEND_BROADCAST = auto()
    SCHEDULE_BROADCAST = auto()

    # System
    VIEW_SETTINGS = auto()
    CONFIGURE_SETTINGS = auto()
    MAINTENANCE_MODE = auto()
    SYSTEM_RESTART = auto()


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[AdminRole, Set[Permission]] = {
    AdminRole.SUPER_ADMIN: set(Permission),  # All permissions

    AdminRole.ADMIN: {
        Permission.VIEW_USERS, Permission.MANAGE_USERS, Permission.BAN_USERS,
        Permission.UNBAN_USERS, Permission.VIEW_ADMINS,
        Permission.VIEW_AI_STATUS, Permission.CONFIGURE_AI, Permission.BENCHMARK_AI,
        Permission.VIEW_DASHBOARD, Permission.VIEW_LOGS, Permission.VIEW_METRICS,
        Permission.VIEW_ERRORS, Permission.SEND_BROADCAST, Permission.SCHEDULE_BROADCAST,
        Permission.VIEW_SETTINGS, Permission.CONFIGURE_SETTINGS,
    },

    AdminRole.OPERATOR: {
        Permission.VIEW_USERS, Permission.MANAGE_USERS, Permission.BAN_USERS,
        Permission.UNBAN_USERS, Permission.VIEW_AI_STATUS, Permission.VIEW_DASHBOARD,
        Permission.VIEW_METRICS, Permission.SEND_BROADCAST,
    },

    AdminRole.MODERATOR: {
        Permission.VIEW_USERS, Permission.BAN_USERS, Permission.UNBAN_USERS,
        Permission.VIEW_DASHBOARD,
    },

    AdminRole.VIEWER: {
        Permission.VIEW_USERS, Permission.VIEW_AI_STATUS, Permission.VIEW_DASHBOARD,
        Permission.VIEW_METRICS,
    },
}


@dataclass
class AdminUser:
    """Admin user data."""
    user_id: int
    role: AdminRole = AdminRole.VIEWER
    custom_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        """Check if admin has a specific permission."""
        # Denied permissions take precedence
        if permission in self.denied_permissions:
            return False

        # Check custom permissions
        if permission in self.custom_permissions:
            return True

        # Check role permissions
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def has_any_permission(self, *permissions: Permission) -> bool:
        """Check if admin has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, *permissions: Permission) -> bool:
        """Check if admin has all of the specified permissions."""
        return all(self.has_permission(p) for p in permissions)


class PermissionManager:
    """Manages admin permissions."""

    def __init__(self) -> None:
        self._admins: Dict[int, AdminUser] = {}

    async def get_admin(self, user_id: int) -> Optional[AdminUser]:
        """Get admin user by ID."""
        return self._admins.get(user_id)

    async def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        return user_id in self._admins

    async def add_admin(
        self,
        user_id: int,
        role: AdminRole = AdminRole.VIEWER,
    ) -> AdminUser:
        """Add a new admin."""
        admin = AdminUser(user_id=user_id, role=role)
        self._admins[user_id] = admin
        logger.info("admin_added", user_id=user_id, role=role.name)
        return admin

    async def remove_admin(self, user_id: int) -> bool:
        """Remove an admin."""
        if user_id in self._admins:
            del self._admins[user_id]
            logger.info("admin_removed", user_id=user_id)
            return True
        return False

    async def set_role(self, user_id: int, role: AdminRole) -> bool:
        """Set admin role."""
        admin = await self.get_admin(user_id)
        if admin:
            admin.role = role
            logger.info("admin_role_changed", user_id=user_id, role=role.name)
            return True
        return False

    async def grant_permission(
        self,
        user_id: int,
        permission: Permission,
    ) -> bool:
        """Grant a custom permission to admin."""
        admin = await self.get_admin(user_id)
        if admin:
            admin.custom_permissions.add(permission)
            admin.denied_permissions.discard(permission)
            return True
        return False

    async def revoke_permission(
        self,
        user_id: int,
        permission: Permission,
    ) -> bool:
        """Revoke a permission from admin."""
        admin = await self.get_admin(user_id)
        if admin:
            admin.denied_permissions.add(permission)
            admin.custom_permissions.discard(permission)
            return True
        return False

    async def check_permission(
        self,
        user_id: int,
        permission: Permission,
    ) -> bool:
        """Check if admin has permission."""
        admin = await self.get_admin(user_id)
        if admin:
            return admin.has_permission(permission)
        return False


# Global permission manager instance
permission_manager = PermissionManager()


def require_admin(func: Callable) -> Callable:
    """Decorator to require admin access."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract user_id from args/kwargs
        user_id = kwargs.get("user_id")
        if user_id is None and len(args) > 0:
            user_id = args[0]

        if not await permission_manager.is_admin(user_id):
            logger.warning("admin_access_denied", user_id=user_id)
            raise PermissionError("Admin access required")

        return await func(*args, **kwargs)

    return wrapper


def require_permission(*permissions: Permission) -> Callable:
    """Decorator to require specific permissions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            user_id = kwargs.get("user_id")
            if user_id is None and len(args) > 0:
                user_id = args[0]

            admin = await permission_manager.get_admin(user_id)
            if not admin:
                raise PermissionError("Admin access required")

            if not admin.has_any_permission(*permissions):
                logger.warning(
                    "permission_denied",
                    user_id=user_id,
                    required=permissions,
                )
                raise PermissionError(f"Required permission: {permissions}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
