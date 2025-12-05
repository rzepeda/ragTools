"""Caching implementation for embeddings."""

from typing import Optional, List, Dict, Any
import time
from collections import OrderedDict
import threading


class EmbeddingCache:
    """In-memory LRU cache for embeddings.

    Thread-safe implementation using an OrderedDict to maintain
    insertion order and implement LRU eviction policy.

    Attributes:
        max_size: Maximum number of embeddings to cache
        ttl: Time-to-live for cache entries in seconds
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding cache.

        Args:
            config: Cache configuration with 'max_size' and 'ttl' keys
        """
        self.max_size = config.get("max_size", 10000)
        self.ttl = config.get("ttl", 3600)  # seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            key: Cache key (typically hash of text + model)

        Returns:
            Embedding vector if found and not expired, None otherwise
        """
        with self._lock:
            # Check if key exists and not expired
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]

            self._misses += 1
            return None

    def set(self, key: str, embedding: List[float]):
        """Set embedding in cache.

        Args:
            key: Cache key
            embedding: Embedding vector to cache
        """
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = embedding
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including size, hits, misses, and hit rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }
