"""User and profile data models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class UserProfile(BaseModel):
    """User profile and preferences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    preferred_language: str = "en"
    timezone: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

    # Privacy and safety
    conversation_history_enabled: bool = True
    data_retention_days: int = 30


class User(BaseModel):
    """User model with metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # Telegram user ID
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    profile: UserProfile = Field(default_factory=UserProfile)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

    total_messages: int = 0

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_activity(self) -> None:
        """Update last active timestamp and increment message count."""
        self.last_active = datetime.utcnow()
        self.total_messages += 1
