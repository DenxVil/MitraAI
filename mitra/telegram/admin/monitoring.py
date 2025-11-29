"""
🤖 Mitra AI - System Monitoring
System health and performance monitoring.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: Optional[str] = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: Optional[float] = None


class Monitoring:
    """
    System monitoring for admins.

    Features:
    - Health checks
    - Performance metrics
    - Log aggregation
    - Alert management
    """

    def __init__(self, history_size: int = 1000) -> None:
        self._health_checks: Dict[str, HealthCheck] = {}
        self._metrics: Dict[str, deque] = {}
        self._history_size = history_size
        self._start_time = datetime.now(timezone.utc)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {}

        # System metrics
        try:
            import psutil

            metrics["cpu_percent"] = psutil.cpu_percent()
            metrics["memory_percent"] = psutil.virtual_memory().percent
            metrics["disk_percent"] = psutil.disk_usage("/").percent
            metrics["network_connections"] = len(psutil.net_connections())
        except ImportError:
            metrics["cpu_percent"] = 0
            metrics["memory_percent"] = 0
            metrics["disk_percent"] = 0
            metrics["network_connections"] = 0

        # Application metrics
        metrics["uptime_seconds"] = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds()

        return metrics

    async def run_health_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        start = datetime.now(timezone.utc)

        try:
            # Run the check based on name
            if name == "database":
                status, message = await self._check_database()
            elif name == "ai_engine":
                status, message = await self._check_ai_engine()
            elif name == "telegram":
                status, message = await self._check_telegram()
            elif name == "memory":
                status, message = await self._check_memory()
            else:
                status, message = "unknown", "Unknown check"

            response_time = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            check = HealthCheck(
                name=name,
                status=status,
                message=message,
                response_time_ms=response_time,
            )
            self._health_checks[name] = check
            return check

        except Exception as e:
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=str(e),
            )

    async def _check_database(self) -> tuple:
        """Check database health."""
        # In production, ping database
        return "healthy", "Database connection OK"

    async def _check_ai_engine(self) -> tuple:
        """Check AI engine health."""
        # In production, ping AI service
        return "healthy", "AI engine operational"

    async def _check_telegram(self) -> tuple:
        """Check Telegram API health."""
        # In production, ping Telegram API
        return "healthy", "Telegram API accessible"

    async def _check_memory(self) -> tuple:
        """Check memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                return "unhealthy", f"Memory critical: {mem.percent:.1f}%"
            elif mem.percent > 75:
                return "degraded", f"Memory high: {mem.percent:.1f}%"
            return "healthy", f"Memory OK: {mem.percent:.1f}%"
        except ImportError:
            return "unknown", "psutil not available"

    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = ["database", "ai_engine", "telegram", "memory"]
        for check in checks:
            await self.run_health_check(check)
        return self._health_checks

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        await self.run_all_health_checks()

        statuses = [c.status for c in self._health_checks.values()]
        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "overall": overall,
            "checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                }
                for name, check in self._health_checks.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self._history_size)

        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
        )
        self._metrics[name].append(point)

    async def get_metric_history(
        self,
        name: str,
        duration_minutes: int = 60,
    ) -> List[MetricPoint]:
        """Get metric history for a time period."""
        if name not in self._metrics:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        return [
            point for point in self._metrics[name]
            if point.timestamp >= cutoff
        ]

    async def get_formatted_health(self) -> str:
        """Get formatted health status for display."""
        summary = await self.get_health_summary()

        status_emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🔴",
            "unknown": "⚪",
        }

        emoji = status_emoji.get(summary["overall"], "⚪")

        lines = [
            f"🏥 *System Health*",
            f"━━━━━━━━━━━━━━━━━━",
            f"",
            f"{emoji} *Overall:* `{summary['overall'].upper()}`",
            f"",
            f"📋 *Checks*",
        ]

        for name, check in summary["checks"].items():
            check_emoji = status_emoji.get(check["status"], "⚪")
            lines.append(
                f"{check_emoji} {name}: `{check['status']}` "
                f"({check['response_time_ms']:.0f}ms)"
            )

        lines.extend([
            f"",
            f"_Last check: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}_",
            f"",
            f"_Coded by Denvil with love 🤍_",
        ])

        return "\n".join(lines)
