"""
Cache implementation for reranking results.

This module provides caching functionality to store and retrieve reranking results,
improving performance for repeated queries.
"""

import time
import hashlib
from typing import Dict, Optional
from .base import RerankResponse, RerankConfig


class RerankCache:
    """
    Cache for storing reranking results with TTL support.

    Example:
        >>> config = RerankConfig(cache_ttl=3600)
        >>> cache = RerankCache(config)
        >>> cache.set("query_hash", response)
        >>> cached = cache.get("query_hash")
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize cache with configuration.

        Args:
            config: Reranking configuration containing cache settings
        """
        self.config = config
        self._cache: Dict[str, tuple[RerankResponse, float]] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }

    def get(self, key: str) -> Optional[RerankResponse]:
        """
        Retrieve a cached reranking response.

        Args:
            key: Cache key (typically a hash of query and documents)

        Returns:
            Cached RerankResponse if found and not expired, None otherwise
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        response, timestamp = self._cache[key]
        current_time = time.time()

        # Check if expired
        if current_time - timestamp > self.config.cache_ttl:
            del self._cache[key]
            self._stats["misses"] += 1
            self._stats["evictions"] += 1
            return None

        self._stats["hits"] += 1
        return response

    def set(self, key: str, response: RerankResponse) -> None:
        """
        Store a reranking response in cache.

        Args:
            key: Cache key
            response: RerankResponse to cache
        """
        self._cache[key] = (response, time.time())
        self._stats["sets"] += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def evict_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries evicted
        """
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self.config.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1

        return len(expired_keys)

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": hit_rate
        }

    def __len__(self) -> int:
        """Return the number of entries in cache."""
        return len(self._cache)

    @staticmethod
    def compute_key(query: str, document_ids: list[str], model_name: str) -> str:
        """
        Compute a cache key from query, documents, and model.

        Args:
            query: Search query
            document_ids: List of document IDs
            model_name: Name of the reranking model

        Returns:
            SHA256 hash as cache key
        """
        doc_ids_str = ",".join(sorted(document_ids))
        content = f"{query}:{doc_ids_str}:{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()
