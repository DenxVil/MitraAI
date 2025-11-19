"""Tests for emotion analyzer."""

import pytest
from mitra.core.emotion_analyzer import EmotionAnalyzer


class TestEmotionAnalyzer:
    """Test cases for EmotionAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EmotionAnalyzer()

    def test_positive_sentiment(self):
        """Test detection of positive sentiment."""
        message = "I'm so happy and excited about this!"
        result = self.analyzer.analyze(message)

        assert result.sentiment == "positive"
        assert "joy" in result.emotions or "excitement" in result.emotions

    def test_negative_sentiment(self):
        """Test detection of negative sentiment."""
        message = "I'm feeling really sad and upset today."
        result = self.analyzer.analyze(message)

        assert result.sentiment == "negative"
        assert "sadness" in result.emotions or "anger" in result.emotions

    def test_neutral_sentiment(self):
        """Test detection of neutral sentiment."""
        message = "What's the weather like today?"
        result = self.analyzer.analyze(message)

        assert result.sentiment == "neutral"

    def test_stress_detection(self):
        """Test detection of stress."""
        message = "I'm so stressed and overwhelmed with everything."
        result = self.analyzer.analyze(message)

        assert "stress" in result.emotions
        assert result.needs_support is True

    def test_crisis_detection(self):
        """Test crisis detection."""
        message = "I want to kill myself"

        is_crisis = self.analyzer.is_crisis(message)
        assert is_crisis is True

        result = self.analyzer.analyze(message)
        assert result.urgency_level == "critical"

    def test_intensity_calculation(self):
        """Test emotional intensity calculation."""
        low_intensity = self.analyzer.analyze("I'm okay")
        high_intensity = self.analyzer.analyze("I'm EXTREMELY angry!!!")

        assert high_intensity.intensity > low_intensity.intensity

    def test_support_needed(self):
        """Test detection of need for support."""
        message = "I'm feeling really anxious and scared about the future."
        result = self.analyzer.analyze(message)

        assert result.needs_support is True
        assert result.intensity >= 0.4
