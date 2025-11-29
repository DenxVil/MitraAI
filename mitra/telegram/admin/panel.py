"""
🤖 Mitra AI - Admin Panel
Main admin controller with dashboard access.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any
from telegram import InlineKeyboardMarkup

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .permissions import (
    AdminRole,
    Permission,
    PermissionManager,
    permission_manager,
)
from ..keyboards import InlineKeyboardBuilder


class AdminPanel:
    """
    Main admin panel controller.

    Features:
    - Admin authentication
    - Role-based menu generation
    - Dashboard access
    - Quick actions
    """

    def __init__(self) -> None:
        self._permission_manager = permission_manager

    async def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        return await self._permission_manager.is_admin(user_id)

    async def get_admin_role(self, user_id: int) -> Optional[AdminRole]:
        """Get admin role for a user."""
        admin = await self._permission_manager.get_admin(user_id)
        if admin:
            return admin.role
        return None

    async def get_main_menu(self, user_id: int) -> InlineKeyboardMarkup:
        """Get the main admin menu based on user permissions."""
        admin = await self._permission_manager.get_admin(user_id)
        if not admin:
            return InlineKeyboardBuilder().build()

        builder = InlineKeyboardBuilder()

        # Dashboard - always available
        if admin.has_permission(Permission.VIEW_DASHBOARD):
            builder.add_button("📊 Dashboard", callback_data="admin:dashboard")

        # User management
        if admin.has_any_permission(Permission.VIEW_USERS, Permission.MANAGE_USERS):
            builder.add_button("👥 Users", callback_data="admin:users")

        builder.new_row()

        # AI control
        if admin.has_any_permission(Permission.VIEW_AI_STATUS, Permission.CONFIGURE_AI):
            builder.add_button("🧠 AI Control", callback_data="admin:ai_control")

        # Monitoring
        if admin.has_any_permission(Permission.VIEW_LOGS, Permission.VIEW_METRICS):
            builder.add_button("📈 Monitoring", callback_data="admin:monitoring")

        builder.new_row()

        # Broadcast
        if admin.has_permission(Permission.SEND_BROADCAST):
            builder.add_button("📢 Broadcast", callback_data="admin:broadcast")

        # Settings
        if admin.has_any_permission(Permission.VIEW_SETTINGS, Permission.CONFIGURE_SETTINGS):
            builder.add_button("⚙️ Settings", callback_data="admin:settings")

        builder.new_row()

        # Close button
        builder.add_button("❌ Close", callback_data="cancel:")

        return builder.build()

    async def get_dashboard_stats(self) -> str:
        """Get formatted dashboard statistics."""
        from .dashboard import Dashboard

        dashboard = Dashboard()
        stats = await dashboard.get_stats()

        return f"""
📊 *System Dashboard*

👥 *Users*
• Total: {stats.get('total_users', 0):,}
• Active (24h): {stats.get('active_users_24h', 0):,}
• New today: {stats.get('new_users_today', 0):,}

💬 *Messages*
• Today: {stats.get('messages_today', 0):,}
• This week: {stats.get('messages_week', 0):,}

🧠 *AI Status*
• Model: {stats.get('model_name', 'N/A')}
• Status: {stats.get('ai_status', 'Unknown')}
• Avg response time: {stats.get('avg_response_time', 'N/A')}

💻 *System*
• CPU: {stats.get('cpu_usage', 'N/A')}%
• Memory: {stats.get('memory_usage', 'N/A')}%
• Uptime: {stats.get('uptime', 'N/A')}

_Last updated: {stats.get('last_updated', 'N/A')}_
"""

    async def handle_action(
        self,
        user_id: int,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle an admin panel action."""
        admin = await self._permission_manager.get_admin(user_id)
        if not admin:
            return {"success": False, "error": "Not an admin"}

        params = params or {}

        if action == "dashboard":
            if not admin.has_permission(Permission.VIEW_DASHBOARD):
                return {"success": False, "error": "Permission denied"}
            stats = await self.get_dashboard_stats()
            return {"success": True, "data": stats}

        elif action == "users":
            if not admin.has_any_permission(Permission.VIEW_USERS):
                return {"success": False, "error": "Permission denied"}
            from .user_management import UserManagement
            mgmt = UserManagement()
            users = await mgmt.list_users(limit=10)
            return {"success": True, "data": users}

        elif action == "ai_control":
            if not admin.has_permission(Permission.VIEW_AI_STATUS):
                return {"success": False, "error": "Permission denied"}
            from .ai_control import AIControl
            control = AIControl()
            status = await control.get_status()
            return {"success": True, "data": status}

        elif action == "monitoring":
            if not admin.has_permission(Permission.VIEW_METRICS):
                return {"success": False, "error": "Permission denied"}
            from .monitoring import Monitoring
            monitoring = Monitoring()
            metrics = await monitoring.get_metrics()
            return {"success": True, "data": metrics}

        elif action == "broadcast":
            if not admin.has_permission(Permission.SEND_BROADCAST):
                return {"success": False, "error": "Permission denied"}
            return {"success": True, "data": "broadcast_ready"}

        else:
            return {"success": False, "error": "Unknown action"}

    async def add_admin(
        self,
        user_id: int,
        role: AdminRole = AdminRole.VIEWER,
    ) -> bool:
        """Add a new admin."""
        await self._permission_manager.add_admin(user_id, role)
        logger.info("admin_added_via_panel", user_id=user_id, role=role.name)
        return True

    async def remove_admin(self, user_id: int) -> bool:
        """Remove an admin."""
        result = await self._permission_manager.remove_admin(user_id)
        if result:
            logger.info("admin_removed_via_panel", user_id=user_id)
        return result

    async def set_admin_role(
        self,
        user_id: int,
        role: AdminRole,
    ) -> bool:
        """Set admin role."""
        result = await self._permission_manager.set_role(user_id, role)
        if result:
            logger.info("admin_role_updated", user_id=user_id, role=role.name)
        return result
