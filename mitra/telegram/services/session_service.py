"""
🤖 Mitra AI - Session Service
Handles conversation sessions and context.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from uuid import uuid4

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: int = 0
    chat_id: int = 0
    messages: List[ConversationMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    thinking_mode: str = "standard"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class SessionService:
    """
    Service for managing conversation sessions.

    Features:
    - Session creation and management
    - Conversation history tracking
    - Context persistence
    - Session expiration
    """

    def __init__(
        self,
        max_messages: int = 50,
        session_timeout_hours: int = 24,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[int, str] = {}  # user_id -> session_id
        self._max_messages = max_messages
        self._session_timeout = timedelta(hours=session_timeout_hours)

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session:
            # Check if session expired
            if datetime.now(timezone.utc) - session.last_active > self._session_timeout:
                await self.close_session(session_id)
                return None
            session.last_active = datetime.now(timezone.utc)
        return session

    async def get_user_session(self, user_id: int) -> Optional[Session]:
        """Get the active session for a user."""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            return await self.get_session(session_id)
        return None

    async def create_session(
        self,
        user_id: int,
        chat_id: int,
        thinking_mode: str = "standard",
    ) -> Session:
        """Create a new session for a user."""
        # Close existing session if any
        existing = await self.get_user_session(user_id)
        if existing:
            await self.close_session(existing.session_id)

        session = Session(
            user_id=user_id,
            chat_id=chat_id,
            thinking_mode=thinking_mode,
        )
        self._sessions[session.session_id] = session
        self._user_sessions[user_id] = session.session_id

        logger.info(
            "session_created",
            session_id=session.session_id,
            user_id=user_id,
        )

        return session

    async def get_or_create_session(
        self,
        user_id: int,
        chat_id: int,
    ) -> Session:
        """Get existing session or create a new one."""
        session = await self.get_user_session(user_id)
        if session:
            return session
        return await self.create_session(user_id, chat_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationMessage]:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if not session:
            return None

        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.messages.append(message)

        # Trim old messages if needed
        if len(session.messages) > self._max_messages:
            session.messages = session.messages[-self._max_messages:]

        return message

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """Get messages from a session."""
        session = await self.get_session(session_id)
        if not session:
            return []

        messages = session.messages
        if limit:
            messages = messages[-limit:]
        return messages

    async def get_context(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get session context."""
        session = await self.get_session(session_id)
        if session:
            return session.context
        return {}

    async def set_context(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Set a context value."""
        session = await self.get_session(session_id)
        if session:
            session.context[key] = value
            return True
        return False

    async def clear_context(self, session_id: str) -> bool:
        """Clear session context."""
        session = await self.get_session(session_id)
        if session:
            session.context.clear()
            return True
        return False

    async def set_thinking_mode(
        self,
        session_id: str,
        mode: str,
    ) -> bool:
        """Set the thinking mode for a session."""
        session = await self.get_session(session_id)
        if session:
            session.thinking_mode = mode
            return True
        return False

    async def close_session(self, session_id: str) -> bool:
        """Close a session."""
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            # Remove from user mapping
            if session.user_id in self._user_sessions:
                del self._user_sessions[session.user_id]
            del self._sessions[session_id]

            logger.info(
                "session_closed",
                session_id=session_id,
                user_id=session.user_id,
                message_count=len(session.messages),
            )
            return True
        return False

    async def clear_messages(self, session_id: str) -> bool:
        """Clear all messages in a session."""
        session = await self.get_session(session_id)
        if session:
            session.messages.clear()
            return True
        return False

    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len([s for s in self._sessions.values() if s.is_active])

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_active > self._session_timeout
        ]
        for sid in expired:
            await self.close_session(sid)

        if expired:
            logger.info("expired_sessions_cleaned", count=len(expired))

        return len(expired)
