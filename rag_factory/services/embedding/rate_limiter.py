"""Rate limiting implementation for embedding service."""

import time
import threading
from typing import Dict, Any


class RateLimiter:
    """Token bucket rate limiter.

    Thread-safe implementation that ensures requests don't exceed
    configured rate limits.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        requests_per_second: Maximum requests allowed per second
        min_interval: Minimum interval between requests in seconds
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the rate limiter.

        Args:
            config: Rate limiter configuration with optional
                   'requests_per_minute' and 'requests_per_second' keys
        """
        self.requests_per_minute = config.get("requests_per_minute", 60)
        self.requests_per_second = config.get("requests_per_second", None)

        # Use most restrictive limit
        if self.requests_per_second:
            self.min_interval = 1.0 / self.requests_per_second
        else:
            self.min_interval = 60.0 / self.requests_per_minute

        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded.

        Blocks the current thread if making a request now would
        exceed the rate limit.
        """
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()
