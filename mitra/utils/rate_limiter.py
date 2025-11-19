"""
Rate limiting implementation for Mitra AI.

Prevents abuse by limiting the number of requests per user within a time window.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict
from .logger import get_logger


logger = get_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter for user requests.

    Note: This is a basic implementation. For production with multiple
    instances, consider using Redis or similar distributed cache.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # Track requests per user: {user_id: [timestamp, timestamp, ...]}
        self._user_requests: Dict[str, list[datetime]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a user is allowed to make a request.

        Args:
            user_id: The user identifier

        Returns:
            True if the request is allowed, False if rate limited
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Clean up old requests outside the window
        self._user_requests[user_id] = [
            timestamp for timestamp in self._user_requests[user_id] if timestamp > cutoff
        ]

        # Check if under the limit
        if len(self._user_requests[user_id]) >= self.max_requests:
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                request_count=len(self._user_requests[user_id]),
                max_requests=self.max_requests,
            )
            return False

        # Add current request
        self._user_requests[user_id].append(now)
        return True

    def get_remaining_requests(self, user_id: str) -> int:
        """
        Get the number of remaining requests for a user.

        Args:
            user_id: The user identifier

        Returns:
            Number of remaining requests in the current window
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Count requests in the current window
        recent_requests = [
            timestamp for timestamp in self._user_requests[user_id] if timestamp > cutoff
        ]

        return max(0, self.max_requests - len(recent_requests))

    def get_time_until_reset(self, user_id: str) -> Optional[int]:
        """
        Get seconds until the rate limit resets for a user.

        Args:
            user_id: The user identifier

        Returns:
            Seconds until reset, or None if no rate limit active
        """
        if not self._user_requests[user_id]:
            return None

        oldest_request = min(self._user_requests[user_id])
        reset_time = oldest_request + timedelta(seconds=self.window_seconds)
        now = datetime.utcnow()

        if reset_time > now:
            return int((reset_time - now).total_seconds())
        return 0

    def clear_user(self, user_id: str) -> None:
        """
        Clear rate limit data for a specific user.

        Args:
            user_id: The user identifier
        """
        if user_id in self._user_requests:
            del self._user_requests[user_id]
            logger.info("rate_limit_cleared", user_id=user_id)
