"""Tests for data models."""

import pytest
from mitra.models import (
    Message, MessageRole, Conversation, EmotionalContext,
    User, UserProfile
)


class TestMessage:
    """Test cases for Message model."""
    
    def test_create_message(self):
        """Test message creation."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, Mitra!"
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, Mitra!"
        assert msg.id is not None
        assert msg.timestamp is not None
    
    def test_message_with_emotion(self):
        """Test message with emotional context."""
        emotion = EmotionalContext(
            sentiment="positive",
            emotions=["joy"],
            intensity=0.8
        )
        
        msg = Message(
            role=MessageRole.USER,
            content="I'm so happy!",
            emotional_context=emotion
        )
        
        assert msg.emotional_context is not None
        assert msg.emotional_context.sentiment == "positive"


class TestConversation:
    """Test cases for Conversation model."""
    
    def test_create_conversation(self):
        """Test conversation creation."""
        conv = Conversation(user_id="user123")
        
        assert conv.user_id == "user123"
        assert len(conv.messages) == 0
        assert conv.id is not None
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation(user_id="user123")
        
        conv.add_message(MessageRole.USER, "Hello")
        conv.add_message(MessageRole.ASSISTANT, "Hi there!")
        
        assert len(conv.messages) == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT
    
    def test_get_recent_messages(self):
        """Test getting recent messages with limit."""
        conv = Conversation(user_id="user123")
        
        # Add 10 messages
        for i in range(10):
            conv.add_message(MessageRole.USER, f"Message {i}")
        
        recent = conv.get_recent_messages(limit=5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 9"
    
    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        conv = Conversation(user_id="user123")
        conv.add_message(MessageRole.USER, "Hello")
        conv.add_message(MessageRole.ASSISTANT, "Hi!")
        
        openai_msgs = conv.to_openai_format()
        
        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[0]["content"] == "Hello"
        assert openai_msgs[1]["role"] == "assistant"


class TestUser:
    """Test cases for User model."""
    
    def test_create_user(self):
        """Test user creation."""
        user = User(
            id="123",
            username="testuser",
            first_name="Test"
        )
        
        assert user.id == "123"
        assert user.username == "testuser"
        assert user.total_messages == 0
    
    def test_update_activity(self):
        """Test updating user activity."""
        user = User(id="123")
        initial_count = user.total_messages
        
        user.update_activity()
        
        assert user.total_messages == initial_count + 1
        assert user.last_active is not None


class TestEmotionalContext:
    """Test cases for EmotionalContext model."""
    
    def test_create_emotional_context(self):
        """Test emotional context creation."""
        context = EmotionalContext(
            sentiment="positive",
            emotions=["joy", "excitement"],
            intensity=0.8,
            needs_support=False,
            urgency_level="low"
        )
        
        assert context.sentiment == "positive"
        assert len(context.emotions) == 2
        assert context.intensity == 0.8
        assert context.needs_support is False
    
    def test_default_values(self):
        """Test default values for emotional context."""
        context = EmotionalContext()
        
        assert context.sentiment == "neutral"
        assert context.emotions == []
        assert context.intensity == 0.5
        assert context.urgency_level == "low"
