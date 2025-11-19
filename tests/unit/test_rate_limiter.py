"""Tests for rate limiter."""

import pytest
import time
from mitra.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter."""

    def test_allows_requests_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        user_id = "test_user"

        # Should allow 5 requests
        for _ in range(5):
            assert limiter.is_allowed(user_id) is True

    def test_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        user_id = "test_user"

        # Use up the limit
        for _ in range(3):
            limiter.is_allowed(user_id)

        # Next request should be blocked
        assert limiter.is_allowed(user_id) is False

    def test_different_users_independent(self):
        """Test that different users have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        user1 = "user1"
        user2 = "user2"

        # User1 uses their limit
        limiter.is_allowed(user1)
        limiter.is_allowed(user1)

        # User2 should still be allowed
        assert limiter.is_allowed(user2) is True

    def test_remaining_requests(self):
        """Test getting remaining request count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        user_id = "test_user"

        assert limiter.get_remaining_requests(user_id) == 5

        limiter.is_allowed(user_id)
        assert limiter.get_remaining_requests(user_id) == 4

        limiter.is_allowed(user_id)
        assert limiter.get_remaining_requests(user_id) == 3

    def test_clear_user(self):
        """Test clearing user rate limit data."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        user_id = "test_user"

        # Use up limit
        limiter.is_allowed(user_id)
        limiter.is_allowed(user_id)
        assert limiter.is_allowed(user_id) is False

        # Clear and try again
        limiter.clear_user(user_id)
        assert limiter.is_allowed(user_id) is True

    def test_time_window_expiry(self):
        """Test that old requests expire from the window."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        user_id = "test_user"

        # Use up limit
        limiter.is_allowed(user_id)
        limiter.is_allowed(user_id)
        assert limiter.is_allowed(user_id) is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed(user_id) is True
