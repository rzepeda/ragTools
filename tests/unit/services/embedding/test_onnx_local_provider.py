"""Unit tests for ONNX local embedding provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def onnx_config():
    """Create ONNX provider configuration."""
    return {"model": "sentence-transformers/all-MiniLM-L6-v2", "max_batch_size": 32}


@pytest.fixture
def mock_onnx_available():
    """Mock ONNX availability."""
    with patch(
        "rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", True
    ):
        yield


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


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_provider_initialization(
    mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test provider initializes correctly."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    # Mock model and tokenizer
    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    provider = ONNXLocalProvider(onnx_config)

    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert provider.get_dimensions() == 384
    assert provider.get_max_batch_size() == 32


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
@patch("rag_factory.services.embedding.providers.onnx_local.torch")
def test_get_embeddings(
    mock_torch, mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test embedding generation."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    # Mock model
    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_outputs = Mock()
    mock_outputs.last_hidden_state = mock_torch.randn(2, 10, 384)
    mock_model.return_value = mock_outputs
    mock_model_class.from_pretrained.return_value = mock_model

    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.tokenize.return_value = ["hello", "world"]
    mock_inputs = {
        "input_ids": mock_torch.tensor([[1, 2, 3]]),
        "attention_mask": mock_torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # Mock torch operations
    mock_tensor = Mock()
    mock_tensor.cpu.return_value.numpy.return_value.tolist.return_value = [
        [0.1] * 384,
        [0.2] * 384,
    ]
    mock_torch.nn.functional.normalize.return_value = mock_tensor

    provider = ONNXLocalProvider(onnx_config)

    # Generate embeddings
    texts = ["hello world", "test text"]
    result = provider.get_embeddings(texts)

    assert result.provider == "onnx-local"
    assert result.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.dimensions == 384
    assert len(result.embeddings) == 2
    assert result.cost == 0.0
    assert result.cached == [False, False]


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_calculate_cost_is_zero(
    mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test that local models have zero cost."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    provider = ONNXLocalProvider(onnx_config)

    cost = provider.calculate_cost(1000)
    assert cost == 0.0


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_known_model_dimensions(
    mock_tokenizer_class, mock_model_class, mock_onnx_available
):
    """Test that known models use predefined dimensions."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 999  # Should be overridden
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    config = {"model": "sentence-transformers/all-mpnet-base-v2"}
    provider = ONNXLocalProvider(config)

    assert provider.get_dimensions() == 768  # From KNOWN_MODELS


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_unknown_model_uses_config(
    mock_tokenizer_class, mock_model_class, mock_onnx_available
):
    """Test that unknown models use model config for dimensions."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 512
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    config = {"model": "some-unknown-model"}
    provider = ONNXLocalProvider(config)

    assert provider.get_dimensions() == 512  # From model.config


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_custom_batch_size(
    mock_tokenizer_class, mock_model_class, mock_onnx_available
):
    """Test custom batch size configuration."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    config = {"model": "test-model", "max_batch_size": 64}
    provider = ONNXLocalProvider(config)

    assert provider.get_max_batch_size() == 64


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_model_loading_failure(
    mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test that model loading failures are handled."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    # Make model loading fail
    mock_model_class.from_pretrained.side_effect = Exception("Model not found")

    with pytest.raises(Exception, match="Model not found"):
        ONNXLocalProvider(onnx_config)


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
def test_get_model_name(
    mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test getting model name."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    provider = ONNXLocalProvider(onnx_config)

    assert provider.get_model_name() == "sentence-transformers/all-MiniLM-L6-v2"


@patch("rag_factory.services.embedding.providers.onnx_local.ORTModelForFeatureExtraction")
@patch("rag_factory.services.embedding.providers.onnx_local.AutoTokenizer")
@patch("rag_factory.services.embedding.providers.onnx_local.torch")
def test_mean_pooling(
    mock_torch, mock_tokenizer_class, mock_model_class, onnx_config, mock_onnx_available
):
    """Test mean pooling implementation."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    mock_model = Mock()
    mock_model.config.hidden_size = 384
    mock_model_class.from_pretrained.return_value = mock_model

    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    provider = ONNXLocalProvider(onnx_config)

    # Create mock tensors
    token_embeddings = Mock()
    attention_mask = Mock()

    # Mock tensor operations
    expanded_mask = Mock()
    attention_mask.unsqueeze.return_value.expand.return_value.float.return_value = (
        expanded_mask
    )

    # Call mean pooling
    result = provider._mean_pooling(token_embeddings, attention_mask)

    # Verify operations were called
    attention_mask.unsqueeze.assert_called_once_with(-1)
    mock_torch.sum.assert_called_once()
    mock_torch.clamp.assert_called_once()
