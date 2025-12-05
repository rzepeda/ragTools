"""Unit tests for embedding service."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.services.embedding.service import EmbeddingService
from rag_factory.services.embedding.config import EmbeddingServiceConfig
from rag_factory.services.embedding.base import EmbeddingResult


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        enable_cache=True,
        enable_rate_limiting=False
    )


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.get_embeddings.return_value = EmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3]] * 2,
        model="test-model",
        dimensions=3,
        token_count=10,
        cost=0.0001,
        provider="mock",
        cached=[False, False]
    )
    provider.get_dimensions.return_value = 3
    provider.get_max_batch_size.return_value = 100
    provider.get_model_name.return_value = "test-model"
    return provider


def test_service_initialization(mock_config):
    """Test service initializes correctly."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider"):
        service = EmbeddingService(mock_config)
        assert service.config == mock_config
        assert service.provider is not None
        assert service.cache is not None


def test_embed_single_text(mock_config, mock_provider):
    """Test embedding single text."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        result = service.embed(["Hello world"])

        assert len(result.embeddings) == 1
        assert result.model == "test-model"
        assert result.dimensions == 3


def test_embed_multiple_texts(mock_config, mock_provider):
    """Test embedding multiple texts."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        result = service.embed(["Text 1", "Text 2"])

        assert len(result.embeddings) == 2
        mock_provider.get_embeddings.assert_called_once()


def test_embed_empty_list_raises_error(mock_config):
    """Test embedding empty list raises error."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider"):
        service = EmbeddingService(mock_config)

        with pytest.raises(ValueError, match="texts cannot be empty"):
            service.embed([])


def test_cache_hit(mock_config, mock_provider):
    """Test cache returns cached embedding."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        # First call - cache miss
        result1 = service.embed(["Hello"])

        # Second call - should hit cache
        result2 = service.embed(["Hello"])

        # Provider should only be called once
        assert mock_provider.get_embeddings.call_count == 1
        assert service._stats["cache_hits"] == 1
        assert service._stats["cache_misses"] == 1


def test_cache_disabled(mock_provider):
    """Test service works with cache disabled."""
    config = EmbeddingServiceConfig(
        provider="openai",
        enable_cache=False
    )

    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(config)
        service.provider = mock_provider

        service.embed(["Hello"])
        service.embed(["Hello"])

        # Provider called twice (no caching)
        assert mock_provider.get_embeddings.call_count == 2


def test_batch_splitting(mock_config, mock_provider):
    """Test automatic batch splitting."""
    mock_provider.get_max_batch_size.return_value = 2

    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        # 5 texts with batch size of 2 should create 3 batches
        result = service.embed(["A", "B", "C", "D", "E"], use_cache=False)

        assert mock_provider.get_embeddings.call_count == 3


def test_get_stats(mock_config, mock_provider):
    """Test statistics tracking."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        service.embed(["Hello"])
        service.embed(["Hello"])  # Cache hit
        service.embed(["World"])  # Cache miss

        stats = service.get_stats()

        assert stats["total_requests"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(0.333, 0.01)


def test_clear_cache(mock_config, mock_provider):
    """Test cache clearing."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        service.embed(["Hello"])
        service.clear_cache()
        service.embed(["Hello"])

        # Should call provider twice (cache cleared)
        assert mock_provider.get_embeddings.call_count == 2


def test_unknown_provider_raises_error():
    """Test unknown provider raises error."""
    # Should raise during config initialization
    with pytest.raises(ValueError, match="Invalid provider"):
        config = EmbeddingServiceConfig(provider="unknown")


def test_compute_cache_key(mock_config, mock_provider):
    """Test cache key computation."""
    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider):
        service = EmbeddingService(mock_config)
        service.provider = mock_provider

        key1 = service._compute_cache_key("Hello")
        key2 = service._compute_cache_key("Hello")
        key3 = service._compute_cache_key("World")

        # Same text should produce same key
        assert key1 == key2
        # Different text should produce different key
        assert key1 != key3
        # Keys should be SHA-256 hashes (64 hex characters)
        assert len(key1) == 64


def test_cache_different_models():
    """Test that different models don't share cache."""
    config1 = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small"
    )
    config2 = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-large"
    )

    mock_provider1 = Mock()
    mock_provider1.get_model_name.return_value = "text-embedding-3-small"
    mock_provider1.get_embeddings.return_value = EmbeddingResult(
        embeddings=[[0.1] * 1536],
        model="text-embedding-3-small",
        dimensions=1536,
        token_count=5,
        cost=0.0001,
        provider="openai",
        cached=[False]
    )
    mock_provider1.get_dimensions.return_value = 1536
    mock_provider1.get_max_batch_size.return_value = 100

    mock_provider2 = Mock()
    mock_provider2.get_model_name.return_value = "text-embedding-3-large"
    mock_provider2.get_embeddings.return_value = EmbeddingResult(
        embeddings=[[0.2] * 3072],
        model="text-embedding-3-large",
        dimensions=3072,
        token_count=5,
        cost=0.0001,
        provider="openai",
        cached=[False]
    )
    mock_provider2.get_dimensions.return_value = 3072
    mock_provider2.get_max_batch_size.return_value = 100

    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider1):
        service1 = EmbeddingService(config1)
        service1.provider = mock_provider1
        key1 = service1._compute_cache_key("test")

    with patch("rag_factory.services.embedding.service.OpenAIProvider", return_value=mock_provider2):
        service2 = EmbeddingService(config2)
        service2.provider = mock_provider2
        key2 = service2._compute_cache_key("test")

    # Different models should produce different cache keys for same text
    assert key1 != key2
