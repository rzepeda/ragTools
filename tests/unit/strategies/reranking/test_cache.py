"""Unit tests for reranking cache."""

import time
import pytest
from rag_factory.strategies.reranking.cache import RerankCache
from rag_factory.strategies.reranking.base import RerankConfig, RerankResponse, RerankResult


@pytest.fixture
def cache_config():
    """Create a cache config with short TTL for testing."""
    return RerankConfig(cache_ttl=1)  # 1 second TTL


@pytest.fixture
def cache(cache_config):
    """Create a cache instance."""
    return RerankCache(cache_config)


@pytest.fixture
def sample_response():
    """Create a sample rerank response."""
    return RerankResponse(
        query="test query",
        results=[
            RerankResult("doc1", 0, 0, 0.9, 0.95, 1.0),
            RerankResult("doc2", 1, 1, 0.8, 0.85, 0.5)
        ],
        total_candidates=2,
        reranked_count=2,
        top_k=10,
        model_used="test-model",
        execution_time_ms=100.0,
        cache_hit=False
    )


class TestRerankCache:
    """Tests for RerankCache."""

    def test_cache_initialization(self, cache_config):
        """Test cache initializes correctly."""
        cache = RerankCache(cache_config)

        assert cache.config == cache_config
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_set_and_get(self, cache, sample_response):
        """Test setting and getting cache entries."""
        cache.set("key1", sample_response)

        assert len(cache) == 1

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.query == sample_response.query
        assert len(retrieved.results) == len(sample_response.results)

    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get("nonexistent")

        assert result is None
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_hit(self, cache, sample_response):
        """Test cache hit."""
        cache.set("key1", sample_response)
        result = cache.get("key1")

        assert result is not None
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_expiration(self, cache, sample_response):
        """Test cache entries expire after TTL."""
        cache.set("key1", sample_response)

        # Should be retrievable immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        result = cache.get("key1")
        assert result is None
        assert len(cache) == 0  # Entry should be removed

    def test_cache_clear(self, cache, sample_response):
        """Test clearing cache."""
        cache.set("key1", sample_response)
        cache.set("key2", sample_response)

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_evict_expired(self, cache, sample_response):
        """Test manual eviction of expired entries."""
        cache.set("key1", sample_response)
        cache.set("key2", sample_response)

        assert len(cache) == 2

        # Wait for expiration
        time.sleep(1.1)

        # Manually evict expired entries
        evicted = cache.evict_expired()

        assert evicted == 2
        assert len(cache) == 0

    def test_cache_stats(self, cache, sample_response):
        """Test cache statistics."""
        # Initially empty
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

        # Add entry and access
        cache.set("key1", sample_response)
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_hit_rate(self, cache, sample_response):
        """Test cache hit rate calculation."""
        cache.set("key1", sample_response)

        # 2 hits, 1 miss = 66.7% hit rate
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(0.667, 0.01)

    def test_compute_key_static(self):
        """Test cache key computation."""
        key1 = RerankCache.compute_key("query", ["doc1", "doc2"], "model1")
        key2 = RerankCache.compute_key("query", ["doc1", "doc2"], "model1")

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        key3 = RerankCache.compute_key("query", ["doc1", "doc3"], "model1")
        assert key1 != key3

        key4 = RerankCache.compute_key("query", ["doc1", "doc2"], "model2")
        assert key1 != key4

        key5 = RerankCache.compute_key("different", ["doc1", "doc2"], "model1")
        assert key1 != key5

    def test_compute_key_document_order(self):
        """Test that document order is normalized in cache key."""
        key1 = RerankCache.compute_key("query", ["doc1", "doc2", "doc3"], "model")
        key2 = RerankCache.compute_key("query", ["doc3", "doc1", "doc2"], "model")

        # Keys should be the same (documents are sorted)
        assert key1 == key2

    def test_multiple_entries(self, cache, sample_response):
        """Test cache with multiple entries."""
        response1 = sample_response
        response2 = RerankResponse(
            query="query2",
            results=[],
            total_candidates=0,
            reranked_count=0,
            top_k=10,
            model_used="test",
            execution_time_ms=0.0,
            cache_hit=False
        )

        cache.set("key1", response1)
        cache.set("key2", response2)

        assert len(cache) == 2
        assert cache.get("key1").query == "test query"
        assert cache.get("key2").query == "query2"

    def test_cache_overwrite(self, cache, sample_response):
        """Test overwriting cache entries."""
        cache.set("key1", sample_response)

        response2 = RerankResponse(
            query="updated query",
            results=[],
            total_candidates=0,
            reranked_count=0,
            top_k=10,
            model_used="test",
            execution_time_ms=0.0,
            cache_hit=False
        )

        cache.set("key1", response2)

        assert len(cache) == 1
        retrieved = cache.get("key1")
        assert retrieved.query == "updated query"
