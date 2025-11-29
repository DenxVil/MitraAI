"""
🤖 Mitra AI - Admin Dashboard
Real-time statistics display for admins.
Coded by Denvil with love 🤍
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class SystemStats:
    """System statistics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    uptime_seconds: int = 0


@dataclass
class AIStats:
    """AI engine statistics."""
    model_name: str = "N/A"
    status: str = "unknown"
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0


@dataclass
class UserStats:
    """User statistics."""
    total_users: int = 0
    active_users_24h: int = 0
    active_users_7d: int = 0
    new_users_today: int = 0
    blocked_users: int = 0


@dataclass
class MessageStats:
    """Message statistics."""
    messages_today: int = 0
    messages_week: int = 0
    messages_month: int = 0
    avg_messages_per_user: float = 0.0


class Dashboard:
    """
    Real-time dashboard for system monitoring.

    Features:
    - System resource monitoring
    - User statistics
    - AI performance metrics
    - Message analytics
    """

    def __init__(self) -> None:
        self._start_time = datetime.now(timezone.utc)
        self._user_stats = UserStats()
        self._message_stats = MessageStats()
        self._ai_stats = AIStats()
        self._system_stats = SystemStats()

    async def get_stats(self) -> Dict[str, Any]:
        """Get all dashboard statistics."""
        await self._refresh_stats()

        return {
            # User stats
            "total_users": self._user_stats.total_users,
            "active_users_24h": self._user_stats.active_users_24h,
            "active_users_7d": self._user_stats.active_users_7d,
            "new_users_today": self._user_stats.new_users_today,
            "blocked_users": self._user_stats.blocked_users,

            # Message stats
            "messages_today": self._message_stats.messages_today,
            "messages_week": self._message_stats.messages_week,
            "messages_month": self._message_stats.messages_month,

            # AI stats
            "model_name": self._ai_stats.model_name,
            "ai_status": self._ai_stats.status,
            "avg_response_time": f"{self._ai_stats.avg_response_time_ms:.0f}ms",
            "ai_error_rate": f"{self._ai_stats.error_rate:.1f}%",

            # System stats
            "cpu_usage": self._system_stats.cpu_usage,
            "memory_usage": self._system_stats.memory_usage,
            "disk_usage": self._system_stats.disk_usage,
            "uptime": self._format_uptime(),

            # Metadata
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def _refresh_stats(self) -> None:
        """Refresh all statistics."""
        await self._refresh_user_stats()
        await self._refresh_message_stats()
        await self._refresh_ai_stats()
        await self._refresh_system_stats()

    async def _refresh_user_stats(self) -> None:
        """Refresh user statistics."""
        # In production, query database
        # For now, use placeholder values
        pass

    async def _refresh_message_stats(self) -> None:
        """Refresh message statistics."""
        # In production, query database
        pass

    async def _refresh_ai_stats(self) -> None:
        """Refresh AI statistics."""
        # In production, query AI service
        self._ai_stats.model_name = "MitraSuperBrain"
        self._ai_stats.status = "operational"

    async def _refresh_system_stats(self) -> None:
        """Refresh system statistics."""
        try:
            import psutil

            self._system_stats.cpu_usage = psutil.cpu_percent()
            self._system_stats.memory_usage = psutil.virtual_memory().percent
            self._system_stats.disk_usage = psutil.disk_usage("/").percent
        except ImportError:
            # psutil not available
            pass

    def _format_uptime(self) -> str:
        """Format uptime string."""
        delta = datetime.now(timezone.utc) - self._start_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

    async def get_user_stats(self) -> UserStats:
        """Get user statistics."""
        await self._refresh_user_stats()
        return self._user_stats

    async def get_message_stats(self) -> MessageStats:
        """Get message statistics."""
        await self._refresh_message_stats()
        return self._message_stats

    async def get_ai_stats(self) -> AIStats:
        """Get AI statistics."""
        await self._refresh_ai_stats()
        return self._ai_stats

    async def get_system_stats(self) -> SystemStats:
        """Get system statistics."""
        await self._refresh_system_stats()
        return self._system_stats

    async def get_formatted_dashboard(self) -> str:
        """Get formatted dashboard text."""
        stats = await self.get_stats()

        return f"""
📊 *Mitra AI Dashboard*
━━━━━━━━━━━━━━━━━━

👥 *Users*
├ Total: `{stats['total_users']:,}`
├ Active (24h): `{stats['active_users_24h']:,}`
├ Active (7d): `{stats['active_users_7d']:,}`
├ New today: `{stats['new_users_today']:,}`
└ Blocked: `{stats['blocked_users']:,}`

💬 *Messages*
├ Today: `{stats['messages_today']:,}`
├ This week: `{stats['messages_week']:,}`
└ This month: `{stats['messages_month']:,}`

🧠 *AI Engine*
├ Model: `{stats['model_name']}`
├ Status: `{stats['ai_status']}`
├ Avg response: `{stats['avg_response_time']}`
└ Error rate: `{stats['ai_error_rate']}`

💻 *System*
├ CPU: `{stats['cpu_usage']:.1f}%`
├ Memory: `{stats['memory_usage']:.1f}%`
├ Disk: `{stats['disk_usage']:.1f}%`
└ Uptime: `{stats['uptime']}`

_Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}_

_Coded by Denvil with love 🤍_
"""
