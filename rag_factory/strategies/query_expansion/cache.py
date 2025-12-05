"""Caching system for query expansion results."""

from typing import Optional, Dict
import time
import logging
from .base import ExpansionResult, ExpansionConfig

logger = logging.getLogger(__name__)


class ExpansionCache:
    """In-memory cache for query expansion results."""

    def __init__(self, config: ExpansionConfig):
        """Initialize cache with configuration.

        Args:
            config: Expansion configuration containing TTL settings
        """
        self.config = config
        self._cache: Dict[str, tuple[ExpansionResult, float]] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }

    def get(self, key: str) -> Optional[ExpansionResult]:
        """Get cached expansion result if valid.

        Args:
            key: Cache key

        Returns:
            Cached ExpansionResult or None if not found or expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        result, timestamp = self._cache[key]
        age = time.time() - timestamp

        # Check if expired
        if age > self.config.cache_ttl:
            del self._cache[key]
            self._stats["evictions"] += 1
            self._stats["misses"] += 1
            logger.debug(f"Cache entry expired: {key} (age: {age:.1f}s)")
            return None

        self._stats["hits"] += 1
        logger.debug(f"Cache hit: {key}")
        return result

    def set(self, key: str, result: ExpansionResult) -> None:
        """Store expansion result in cache.

        Args:
            key: Cache key
            result: Expansion result to cache
        """
        self._cache[key] = (result, time.time())
        self._stats["size"] = len(self._cache)
        logger.debug(f"Cached result: {key}")

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        self._stats["size"] = 0
        logger.info(f"Cache cleared: {count} entries removed")

    def evict_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries evicted
        """
        current_time = time.time()
        expired_keys = []

        for key, (result, timestamp) in self._cache.items():
            age = current_time - timestamp
            if age > self.config.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1

        if expired_keys:
            self._stats["size"] = len(self._cache)
            logger.info(f"Evicted {len(expired_keys)} expired entries")

        return len(expired_keys)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }
