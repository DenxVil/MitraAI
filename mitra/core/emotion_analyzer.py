"""
Emotion analysis and sentiment detection for Mitra AI.

Analyzes user messages to detect emotional state and adapt responses accordingly.
"""

from typing import List, Optional
import re
from ..models import EmotionalContext
from ..utils import get_logger


logger = get_logger(__name__)


class EmotionAnalyzer:
    """
    Analyzes emotional content in user messages.

    Uses pattern matching and keyword analysis for basic emotion detection.
    In production, this could be enhanced with ML models or AI-based analysis.
    """

    # Emotion keyword patterns
    EMOTION_PATTERNS = {
        "joy": [
            "happy",
            "excited",
            "great",
            "wonderful",
            "amazing",
            "love",
            "enjoy",
            "glad",
            "delighted",
        ],
        "sadness": ["sad", "unhappy", "depressed", "down", "blue", "miserable", "gloomy"],
        "anger": ["angry", "mad", "furious", "annoyed", "frustrated", "irritated", "upset"],
        "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified", "fear"],
        "stress": ["stressed", "overwhelmed", "pressure", "burden", "exhausted", "tired"],
        "confusion": ["confused", "lost", "don't understand", "unclear", "puzzled", "bewildered"],
        "gratitude": ["thank", "grateful", "appreciate", "thanks"],
        "excitement": ["excited", "can't wait", "looking forward", "thrilled"],
    }

    # Crisis indicators - serious patterns that need urgent attention
    CRISIS_PATTERNS = [
        r"\b(want to|going to|thinking about)\s+(die|kill myself|end it|suicide)\b",
        r"\bhurt (myself|themselves)\b",
        r"\bno reason to (live|go on)\b",
        r"\bcan't (take it|go on) anymore\b",
    ]

    # Urgency indicators
    URGENCY_PATTERNS = {
        "high": ["urgent", "emergency", "immediate", "right now", "asap", "critical"],
        "medium": ["soon", "important", "need help", "struggling"],
    }

    def analyze(self, message: str) -> EmotionalContext:
        """
        Analyze emotional content of a message.

        Args:
            message: The user's message text

        Returns:
            Emotional context analysis
        """
        message_lower = message.lower()

        # Detect emotions
        detected_emotions = self._detect_emotions(message_lower)

        # Determine sentiment
        sentiment = self._determine_sentiment(detected_emotions)

        # Calculate intensity
        intensity = self._calculate_intensity(message_lower, detected_emotions)

        # Check if support is needed
        needs_support = self._needs_support(detected_emotions, intensity)

        # Determine urgency
        urgency_level = self._determine_urgency(message_lower)

        context = EmotionalContext(
            sentiment=sentiment,
            emotions=detected_emotions,
            intensity=intensity,
            needs_support=needs_support,
            urgency_level=urgency_level,
        )

        logger.debug(
            "emotion_analysis_complete",
            sentiment=sentiment,
            emotions=detected_emotions,
            intensity=intensity,
            needs_support=needs_support,
            urgency=urgency_level,
        )

        return context

    def _detect_emotions(self, message: str) -> List[str]:
        """Detect emotions based on keyword patterns."""
        detected = []

        for emotion, keywords in self.EMOTION_PATTERNS.items():
            if any(keyword in message for keyword in keywords):
                detected.append(emotion)

        return detected if detected else ["neutral"]

    def _determine_sentiment(self, emotions: List[str]) -> str:
        """Determine overall sentiment from detected emotions."""
        positive_emotions = {"joy", "gratitude", "excitement"}
        negative_emotions = {"sadness", "anger", "fear", "stress"}

        positive_count = sum(1 for e in emotions if e in positive_emotions)
        negative_count = sum(1 for e in emotions if e in negative_emotions)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_intensity(self, message: str, emotions: List[str]) -> float:
        """Calculate emotional intensity."""
        # Base intensity on number of emotions and strong language
        intensity = min(len(emotions) * 0.2, 0.6)

        # Check for intensifiers
        intensifiers = ["very", "extremely", "really", "so", "totally", "completely"]
        if any(word in message for word in intensifiers):
            intensity = min(intensity + 0.2, 1.0)

        # Check for strong punctuation
        if "!" in message or message.isupper():
            intensity = min(intensity + 0.1, 1.0)

        return round(intensity, 2)

    def _needs_support(self, emotions: List[str], intensity: float) -> bool:
        """Determine if the user appears to need emotional support."""
        support_emotions = {"sadness", "anger", "fear", "stress", "confusion"}

        # High intensity negative emotions indicate need for support
        if intensity > 0.5 and any(e in support_emotions for e in emotions):
            return True

        # Multiple negative emotions
        if sum(1 for e in emotions if e in support_emotions) >= 2:
            return True

        # Single strong support emotion with reasonable intensity
        if any(e in support_emotions for e in emotions) and intensity >= 0.4:
            return True

        return False

    def _determine_urgency(self, message: str) -> str:
        """Determine urgency level of the message."""
        # Check for crisis patterns first
        for pattern in self.CRISIS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                logger.warning("crisis_pattern_detected", message_preview=message[:100])
                return "critical"

        # Check for urgency keywords
        for level, keywords in self.URGENCY_PATTERNS.items():
            if any(keyword in message for keyword in keywords):
                return level

        return "low"

    def is_crisis(self, message: str) -> bool:
        """
        Check if message contains crisis indicators.

        Args:
            message: The user's message

        Returns:
            True if crisis indicators detected
        """
        for pattern in self.CRISIS_PATTERNS:
            if re.search(pattern, message.lower(), re.IGNORECASE):
                return True
        return False
