"""
Safety and content moderation for Mitra AI.

Handles content filtering, crisis detection, and appropriate boundary setting.
"""

from typing import Optional, Tuple
import re
from ..utils import get_logger


logger = get_logger(__name__)


class SafetyFilter:
    """
    Content safety and moderation filter.

    Detects inappropriate content, crisis situations, and enforces safety boundaries.
    """

    # Topics that require professional help
    PROFESSIONAL_HELP_TOPICS = {
        "suicide": "mental health professional",
        "self-harm": "mental health professional",
        "violence": "appropriate authorities",
        "abuse": "appropriate authorities",
        "medical emergency": "medical professional",
        "legal issue": "legal professional",
    }

    # Harmful content patterns
    HARMFUL_PATTERNS = [
        r"\b(how to|ways to|help me)\s+(kill|harm|hurt)\b",
        r"\b(make|create|build)\s+(bomb|weapon|explosive|pipe\s*bomb)\b",
        r"\bhow\s+to\s+make\s+a\s+bomb\b",
        r"\b(hack|break into|steal)\b",
        r"\b(illegal|unlawful)\s+(drugs|substances)\b",
    ]

    # Crisis support resources
    CRISIS_RESOURCES = """
If you're experiencing a mental health crisis, please reach out to professional help:

🆘 **Immediate Help:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

🌍 **International Resources:**
- Find your country's crisis line: https://findahelpline.com/

Remember: I'm an AI assistant and while I'm here to support you, trained professionals can provide the immediate help you need."""

    def __init__(self, enable_moderation: bool = True):
        """
        Initialize the safety filter.

        Args:
            enable_moderation: Whether to enable content moderation
        """
        self.enable_moderation = enable_moderation

    def check_message(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a message is safe and appropriate.

        Args:
            message: The message to check

        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        if not self.enable_moderation:
            return True, None

        message_lower = message.lower()

        # Check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(
                    "harmful_content_detected", pattern=pattern, message_preview=message[:50]
                )
                return False, "harmful_content"

        return True, None

    def detect_crisis(self, message: str) -> bool:
        """
        Detect if message indicates a crisis situation.

        Args:
            message: The user's message

        Returns:
            True if crisis detected
        """
        message_lower = message.lower()

        crisis_keywords = [
            "suicide",
            "suicidal",
            "kill myself",
            "end my life",
            "want to die",
            "don't want to live",
            "self-harm",
            "hurt myself",
            "no reason to live",
        ]

        for keyword in crisis_keywords:
            if keyword in message_lower:
                logger.warning("crisis_detected", keyword=keyword, message_preview=message[:50])
                return True

        return False

    def get_crisis_response(self) -> str:
        """Get appropriate crisis response with resources."""
        return f"""I'm deeply concerned about what you've shared. While I'm here to listen and support you, I'm an AI and not equipped to provide the immediate help you need right now.

{self.CRISIS_RESOURCES}

Your life matters, and there are people who want to help. Please reach out to one of these services - they're available 24/7 and staffed by trained professionals who can provide real support.

If you're in immediate danger, please call emergency services (911 in the US) or go to your nearest emergency room.

I'm still here if you'd like to talk, but please also connect with professional help."""

    def get_professional_help_response(self, topic: str) -> str:
        """
        Get response recommending professional help.

        Args:
            topic: The topic requiring professional help

        Returns:
            Appropriate response message
        """
        professional_type = self.PROFESSIONAL_HELP_TOPICS.get(topic, "professional")

        return f"""I understand this is important to you, but this is an area where you really need guidance from a {professional_type}.

While I can offer general information and support, I'm not qualified to provide advice on {topic}. A qualified professional can give you the specific, personalized help you need.

Is there anything else I can help you with, or would you like me to provide information about finding appropriate professional help?"""

    def get_boundary_response(self, reason: str = "inappropriate") -> str:
        """
        Get response for when boundaries are exceeded.

        Args:
            reason: Reason for the boundary

        Returns:
            Boundary response message
        """
        responses = {
            "harmful_content": "I can't help with that request as it could cause harm. I'm designed to be helpful, harmless, and honest. Is there something positive I can assist you with instead?",
            "inappropriate": "I'm not able to engage with that type of content. I'm here to have respectful, helpful conversations. How else can I help you today?",
            "illegal": "I can't provide assistance with illegal activities. If you're facing a difficult situation, I'd be happy to help you think through legal and positive solutions.",
        }

        return responses.get(reason, responses["inappropriate"])

    def should_refuse(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if Mitra should refuse to respond.

        Args:
            message: The user's message

        Returns:
            Tuple of (should_refuse, refusal_reason)
        """
        # Check basic safety
        is_safe, reason = self.check_message(message)
        if not is_safe:
            return True, reason

        # Check for crisis (don't refuse, but handle specially)
        if self.detect_crisis(message):
            return False, "crisis"  # Special case - don't refuse but redirect

        return False, None
