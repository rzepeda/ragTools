"""Unit tests for ONNX local embedding provider.

Tests use centralized mock_onnx_env fixture from conftest.py which handles
all ONNX-related mocking automatically.
"""

import pytest
import numpy as np
from unittest.mock import patch


@pytest.fixture
def onnx_config():
    """Create ONNX provider configuration."""
    # Use Xenova model (384 dimensions)
    return {"model": "Xenova/all-MiniLM-L6-v2", "max_batch_size": 32}


# Note: Using centralized mock_onnx_env fixture from conftest.py
# This replaces the need for @patch decorators on every test function


def test_provider_not_available_raises_error():
    """Test that missing ONNX raises informative error."""
    with patch(
        "rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", False
    ):
        from rag_factory.services.embedding.providers.onnx_local import (
            ONNXLocalProvider,
        )

        with pytest.raises(ImportError, match="ONNX Runtime not installed"):
            ONNXLocalProvider({"model": "test"})


def test_provider_initialization(onnx_config, mock_onnx_env):
    """Test provider initializes correctly.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        provider = ONNXLocalProvider(onnx_config)

        assert provider.model_name == "Xenova/all-MiniLM-L6-v2"
        assert provider.get_dimensions() == 384
        assert provider.get_max_batch_size() == 32


def test_get_embeddings(onnx_config, mock_onnx_env):
    """Test embedding generation.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        provider = ONNXLocalProvider(onnx_config)
        texts = ["hello world", "test text"]
        result = provider.get_embeddings(texts)

        assert result.provider == "onnx-local"
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 384
        assert result.model == "Xenova/all-MiniLM-L6-v2"


def test_calculate_cost_is_zero(onnx_config, mock_onnx_env):
    """Test that local ONNX provider has zero cost.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        provider = ONNXLocalProvider(onnx_config)
        cost = provider.calculate_cost(token_count=100)

        assert cost == 0.0


def test_known_model_dimensions(mock_onnx_env):
    """Test that known models return correct dimensions.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        config = {"model": "Xenova/all-MiniLM-L6-v2"}
        provider = ONNXLocalProvider(config)

        assert provider.get_dimensions() == 384


def test_unknown_model_uses_output_shape(mock_onnx_env):
    """Test that unknown models infer dimensions from ONNX output shape.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=768):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        config = {"model": "unknown/custom-model"}
        provider = ONNXLocalProvider(config)

        # Should infer from mocked output shape
        assert provider.get_dimensions() == 768


def test_custom_batch_size(mock_onnx_env):
    """Test custom batch size configuration.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        config = {"model": "Xenova/all-MiniLM-L6-v2", "max_batch_size": 64}
        provider = ONNXLocalProvider(config)

        assert provider.get_max_batch_size() == 64


def test_model_loading_failure():
    """Test that model loading failures are handled gracefully."""
    with patch(
        "rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", True
    ):
        with patch(
            "rag_factory.services.embedding.providers.onnx_local.get_onnx_model_path",
            side_effect=FileNotFoundError("Model not found")
        ):
            from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

            with pytest.raises(FileNotFoundError, match="Model not found"):
                ONNXLocalProvider({"model": "nonexistent/model"})


def test_get_model_name(onnx_config, mock_onnx_env):
    """Test get_model_name returns correct model name.
    
    Uses centralized mock_onnx_env to handle all ONNX mocking.
    """
    with mock_onnx_env(dimension=384):
        from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

        provider = ONNXLocalProvider(onnx_config)

        assert provider.get_model_name() == "Xenova/all-MiniLM-L6-v2"


def test_mean_pooling():
    """Test mean pooling utility function."""
    from rag_factory.services.embedding.providers.onnx_local import mean_pooling

    # Create sample token embeddings and attention mask
    token_embeddings = np.array([
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # Batch 1: 2 real tokens, 1 padding
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]  # Batch 2: 3 real tokens
    ])
    attention_mask = np.array([
        [1, 1, 0],  # Batch 1: mask out padding
        [1, 1, 1]   # Batch 2: all tokens valid
    ])

    result = mean_pooling(token_embeddings, attention_mask)

    # Expected: average of non-masked tokens
    # Batch 1: mean of [1,2] and [3,4] = [2.0, 3.0]
    # Batch 2: mean of [5,6], [7,8], [9,10] = [7.0, 8.0]
    expected = np.array([[2.0, 3.0], [7.0, 8.0]])

    np.testing.assert_array_almost_equal(result, expected)
