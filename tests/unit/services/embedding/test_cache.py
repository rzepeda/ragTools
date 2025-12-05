"""Unit tests for embedding cache."""

import pytest
import time
import threading
from rag_factory.services.embedding.cache import EmbeddingCache


@pytest.fixture
def cache():
    """Create a cache instance for testing."""
    return EmbeddingCache({"max_size": 3, "ttl": 1})


def test_cache_set_and_get(cache):
    """Test basic set and get operations."""
    embedding = [0.1, 0.2, 0.3]
    cache.set("key1", embedding)

    result = cache.get("key1")
    assert result == embedding


def test_cache_miss(cache):
    """Test cache miss returns None."""
    result = cache.get("nonexistent")
    assert result is None


def test_cache_expiration(cache):
    """Test cache entry expires after TTL."""
    embedding = [0.1, 0.2, 0.3]
    cache.set("key1", embedding)

    # Wait for expiration
    time.sleep(1.1)

    result = cache.get("key1")
    assert result is None


def test_cache_max_size(cache):
    """Test cache respects max size."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])
    cache.set("key3", [0.3])
    cache.set("key4", [0.4])  # Should evict key1

    assert cache.get("key1") is None
    assert cache.get("key4") is not None


def test_cache_lru_eviction(cache):
    """Test LRU eviction policy."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])
    cache.set("key3", [0.3])

    # Access key1 to make it most recent
    cache.get("key1")

    # Add new key, should evict key2 (least recently used)
    cache.set("key4", [0.4])

    assert cache.get("key2") is None
    assert cache.get("key1") is not None


def test_cache_clear(cache):
    """Test cache clearing."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_stats(cache):
    """Test cache statistics."""
    cache.set("key1", [0.1])

    cache.get("key1")  # Hit
    cache.get("key2")  # Miss
    cache.get("key1")  # Hit

    stats = cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.666, 0.01)
    assert stats["size"] == 1


def test_cache_thread_safety():
    """Test cache is thread-safe."""
    cache = EmbeddingCache({"max_size": 100, "ttl": 10})

    def worker(thread_id):
        for i in range(100):
            cache.set(f"key_{thread_id}_{i}", [float(i)])
            cache.get(f"key_{thread_id}_{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Should not crash
    stats = cache.get_stats()
    assert stats["size"] <= 100


def test_cache_update_existing_key(cache):
    """Test updating an existing key."""
    cache.set("key1", [0.1, 0.2])
    cache.set("key1", [0.3, 0.4])  # Update

    result = cache.get("key1")
    assert result == [0.3, 0.4]


def test_cache_large_embeddings():
    """Test cache with large embedding vectors."""
    cache = EmbeddingCache({"max_size": 10, "ttl": 60})

    # Create a large embedding (1536 dimensions like OpenAI)
    large_embedding = [0.1] * 1536

    cache.set("large_key", large_embedding)
    result = cache.get("large_key")

    assert len(result) == 1536
    assert result == large_embedding
