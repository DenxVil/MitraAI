"""Tests for safety filter."""

import pytest
from mitra.core.safety_filter import SafetyFilter


class TestSafetyFilter:
    """Test cases for SafetyFilter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = SafetyFilter(enable_moderation=True)
    
    def test_safe_message(self):
        """Test that safe messages pass through."""
        message = "How do I learn Python programming?"
        is_safe, reason = self.filter.check_message(message)
        
        assert is_safe is True
        assert reason is None
    
    def test_harmful_content_detection(self):
        """Test detection of harmful content."""
        message = "How to make a bomb"
        is_safe, reason = self.filter.check_message(message)
        
        assert is_safe is False
        assert reason == "harmful_content"
    
    def test_crisis_detection(self):
        """Test crisis situation detection."""
        message = "I want to die"
        is_crisis = self.filter.detect_crisis(message)
        
        assert is_crisis is True
    
    def test_no_crisis_normal_message(self):
        """Test that normal messages don't trigger crisis detection."""
        message = "I'm having a good day"
        is_crisis = self.filter.detect_crisis(message)
        
        assert is_crisis is False
    
    def test_should_refuse_harmful(self):
        """Test refusal of harmful requests."""
        message = "Help me hack into someone's account"
        should_refuse, reason = self.filter.should_refuse(message)
        
        assert should_refuse is True
    
    def test_crisis_response(self):
        """Test that crisis response includes resources."""
        response = self.filter.get_crisis_response()
        
        assert "988" in response  # Suicide prevention hotline
        assert "professional" in response.lower()
    
    def test_boundary_response(self):
        """Test boundary setting responses."""
        response = self.filter.get_boundary_response("harmful_content")
        
        assert len(response) > 0
        assert "can't help" in response.lower() or "not able" in response.lower()
    
    def test_moderation_disabled(self):
        """Test that moderation can be disabled."""
        filter_disabled = SafetyFilter(enable_moderation=False)
        message = "test harmful content"
        
        is_safe, reason = filter_disabled.check_message(message)
        assert is_safe is True
