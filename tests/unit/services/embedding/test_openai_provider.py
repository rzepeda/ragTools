"""Unit tests for OpenAI provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def openai_config():
    """Create OpenAI provider configuration."""
    return {
        "api_key": "test-key",
        "model": "text-embedding-3-small"
    }


def test_provider_initialization(openai_config):
    """Test provider initializes correctly."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            provider = OpenAIProvider(openai_config)
            assert provider.model == "text-embedding-3-small"
            assert provider.get_dimensions() == 1536


def test_provider_invalid_model():
    """Test provider raises error for invalid model."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            config = {"api_key": "test-key", "model": "invalid-model"}

            with pytest.raises(ValueError, match="Unknown OpenAI model"):
                OpenAIProvider(config)


def test_provider_missing_api_key():
    """Test provider raises error when API key is missing."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            config = {"model": "text-embedding-3-small"}

            with pytest.raises(ValueError, match="API key is required"):
                OpenAIProvider(config)


def test_get_embeddings(openai_config):
    """Test getting embeddings."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai") as mock_openai:
            from rag_factory.services.embedding.providers.openai import OpenAIProvider

            # Mock OpenAI response
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_response.usage.total_tokens = 10
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            provider = OpenAIProvider(openai_config)
            result = provider.get_embeddings(["Hello", "World"])

            assert len(result.embeddings) == 2
            assert result.token_count == 10
            assert result.model == "text-embedding-3-small"
            assert result.provider == "openai"
            assert result.cached == [False, False]


def test_calculate_cost(openai_config):
    """Test cost calculation."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            provider = OpenAIProvider(openai_config)

            # text-embedding-3-small costs $0.00002 per 1k tokens
            cost = provider.calculate_cost(1000)
            assert cost == pytest.approx(0.00002)

            cost = provider.calculate_cost(5000)
            assert cost == pytest.approx(0.0001)


def test_get_max_batch_size(openai_config):
    """Test max batch size."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            provider = OpenAIProvider(openai_config)
            assert provider.get_max_batch_size() == 100


def test_get_model_name(openai_config):
    """Test getting model name."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider
            provider = OpenAIProvider(openai_config)
            assert provider.get_model_name() == "text-embedding-3-small"


def test_different_models():
    """Test different OpenAI models."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", True):
        with patch("rag_factory.services.embedding.providers.openai.openai"):
            from rag_factory.services.embedding.providers.openai import OpenAIProvider

            # Test text-embedding-3-large
            config = {"api_key": "test-key", "model": "text-embedding-3-large"}
            provider = OpenAIProvider(config)
            assert provider.get_dimensions() == 3072

            # Test text-embedding-ada-002
            config = {"api_key": "test-key", "model": "text-embedding-ada-002"}
            provider = OpenAIProvider(config)
            assert provider.get_dimensions() == 1536


def test_openai_not_installed():
    """Test error when OpenAI is not installed."""
    with patch("rag_factory.services.embedding.providers.openai.OPENAI_AVAILABLE", False):
        from rag_factory.services.embedding.providers.openai import OpenAIProvider

        with pytest.raises(ImportError, match="OpenAI package not installed"):
            OpenAIProvider({"api_key": "test"})
