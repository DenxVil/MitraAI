"""
🤖 Mitra AI - Broadcast System
Admin broadcast messaging to users.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import asyncio

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BroadcastStatus(Enum):
    """Broadcast status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class BroadcastMessage:
    """Broadcast message details."""
    message_id: str
    text: str
    parse_mode: str = "Markdown"
    disable_notification: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: Optional[datetime] = None
    status: BroadcastStatus = BroadcastStatus.PENDING


@dataclass
class BroadcastResult:
    """Result of a broadcast operation."""
    message_id: str
    status: BroadcastStatus
    total_users: int = 0
    sent_count: int = 0
    failed_count: int = 0
    blocked_users: Set[int] = field(default_factory=set)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Broadcast:
    """
    Broadcast messaging system for admins.

    Features:
    - Send to all users
    - Send to specific groups
    - Schedule broadcasts
    - Track delivery status
    """

    def __init__(self) -> None:
        self._broadcasts: Dict[str, BroadcastMessage] = {}
        self._results: Dict[str, BroadcastResult] = {}
        self._running_broadcasts: Set[str] = set()

    async def create_broadcast(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
        scheduled_for: Optional[datetime] = None,
    ) -> BroadcastMessage:
        """Create a new broadcast message."""
        from uuid import uuid4

        message_id = str(uuid4())[:8]
        message = BroadcastMessage(
            message_id=message_id,
            text=text,
            parse_mode=parse_mode,
            disable_notification=disable_notification,
            scheduled_for=scheduled_for,
        )
        self._broadcasts[message_id] = message

        logger.info(
            "broadcast_created",
            message_id=message_id,
            scheduled=scheduled_for is not None,
        )

        return message

    async def send_broadcast(
        self,
        message_id: str,
        user_ids: Optional[List[int]] = None,
        send_func: Optional[callable] = None,
    ) -> BroadcastResult:
        """Send a broadcast message."""
        message = self._broadcasts.get(message_id)
        if not message:
            raise ValueError(f"Broadcast {message_id} not found")

        if message_id in self._running_broadcasts:
            raise ValueError(f"Broadcast {message_id} is already running")

        result = BroadcastResult(
            message_id=message_id,
            status=BroadcastStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        self._results[message_id] = result
        self._running_broadcasts.add(message_id)
        message.status = BroadcastStatus.RUNNING

        try:
            # Get user IDs if not provided
            if user_ids is None:
                user_ids = await self._get_all_user_ids()

            result.total_users = len(user_ids)

            # Send to each user
            for user_id in user_ids:
                try:
                    if send_func:
                        await send_func(
                            chat_id=user_id,
                            text=message.text,
                            parse_mode=message.parse_mode,
                            disable_notification=message.disable_notification,
                        )
                    result.sent_count += 1

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.05)

                except Exception as e:
                    result.failed_count += 1
                    if "blocked" in str(e).lower():
                        result.blocked_users.add(user_id)

            result.status = BroadcastStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            message.status = BroadcastStatus.COMPLETED

            logger.info(
                "broadcast_completed",
                message_id=message_id,
                sent=result.sent_count,
                failed=result.failed_count,
            )

        except Exception as e:
            result.status = BroadcastStatus.FAILED
            message.status = BroadcastStatus.FAILED
            logger.error("broadcast_failed", message_id=message_id, error=str(e))

        finally:
            self._running_broadcasts.discard(message_id)

        return result

    async def cancel_broadcast(self, message_id: str) -> bool:
        """Cancel a running broadcast."""
        message = self._broadcasts.get(message_id)
        if not message:
            return False

        if message.status != BroadcastStatus.RUNNING:
            return False

        message.status = BroadcastStatus.CANCELLED
        self._running_broadcasts.discard(message_id)

        if message_id in self._results:
            self._results[message_id].status = BroadcastStatus.CANCELLED

        logger.info("broadcast_cancelled", message_id=message_id)
        return True

    async def get_broadcast_status(
        self,
        message_id: str,
    ) -> Optional[BroadcastResult]:
        """Get broadcast status."""
        return self._results.get(message_id)

    async def list_broadcasts(
        self,
        status: Optional[BroadcastStatus] = None,
    ) -> List[BroadcastMessage]:
        """List broadcasts with optional status filter."""
        broadcasts = list(self._broadcasts.values())
        if status:
            broadcasts = [b for b in broadcasts if b.status == status]
        return broadcasts

    async def _get_all_user_ids(self) -> List[int]:
        """Get all user IDs for broadcast."""
        # In production, query database
        return []

    async def get_formatted_status(
        self,
        message_id: str,
    ) -> str:
        """Get formatted broadcast status for display."""
        result = await self.get_broadcast_status(message_id)
        if not result:
            return "Broadcast not found."

        status_emoji = {
            BroadcastStatus.PENDING: "⏳",
            BroadcastStatus.RUNNING: "🔄",
            BroadcastStatus.COMPLETED: "✅",
            BroadcastStatus.CANCELLED: "⛔",
            BroadcastStatus.FAILED: "❌",
        }

        emoji = status_emoji.get(result.status, "❓")
        progress = (
            f"{result.sent_count}/{result.total_users}"
            if result.total_users > 0
            else "0/0"
        )

        return f"""
📢 *Broadcast Status*
━━━━━━━━━━━━━━━━━━

{emoji} *Status:* `{result.status.value.upper()}`
🆔 *ID:* `{result.message_id}`

📊 *Progress:* `{progress}`
├ ✅ Sent: `{result.sent_count:,}`
├ ❌ Failed: `{result.failed_count:,}`
└ 🚫 Blocked: `{len(result.blocked_users):,}`

⏰ *Timing*
├ Started: `{result.started_at.strftime('%H:%M:%S') if result.started_at else 'N/A'}`
└ Completed: `{result.completed_at.strftime('%H:%M:%S') if result.completed_at else 'N/A'}`

_Coded by Denvil with love 🤍_
"""
