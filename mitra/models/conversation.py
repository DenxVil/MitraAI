"""Conversation and message data models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class EmotionalContext(BaseModel):
    """Emotional context detected from user message."""
    
    sentiment: str = Field(
        default="neutral",
        description="Overall sentiment: positive, negative, neutral"
    )
    emotions: List[str] = Field(
        default_factory=list,
        description="Detected emotions: joy, sadness, anger, fear, stress, etc."
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotional intensity from 0 (calm) to 1 (intense)"
    )
    needs_support: bool = Field(
        default=False,
        description="Whether user appears to need emotional support"
    )
    urgency_level: str = Field(
        default="low",
        description="Urgency level: low, medium, high, critical"
    )


class Message(BaseModel):
    """A single message in a conversation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotional_context: Optional[EmotionalContext] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    """A conversation thread with message history."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, 
                   emotional_context: Optional[EmotionalContext] = None) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            emotional_context=emotional_context
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def to_openai_format(self, max_messages: Optional[int] = None) -> List[dict]:
        """Convert conversation to OpenAI API format."""
        messages = self.get_recent_messages(max_messages) if max_messages else self.messages
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
