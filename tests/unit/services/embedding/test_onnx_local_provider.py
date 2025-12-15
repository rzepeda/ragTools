"""Unit tests for ONNX local embedding provider.

Note: These tests have been updated to work with the lightweight tokenizers library
instead of transformers.AutoTokenizer, matching the actual implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def onnx_config():
    """Create ONNX provider configuration."""
    # Use Xenova model (768 dimensions)
    return {"model": "Xenova/all-mpnet-base-v2", "max_batch_size": 32}


@pytest.fixture
def mock_onnx_available():
    """Mock ONNX availability."""
    with patch(
        "rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", True
    ):
        yield


@pytest.fixture
def mock_hf_hub_available():
    """Mock HuggingFace Hub availability."""
    with patch(
        "rag_factory.services.embedding.providers.onnx_local.HF_HUB_AVAILABLE", True
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


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_provider_initialization(
    mock_metadata, mock_validate, mock_download, mock_session, onnx_config, mock_onnx_available, mock_hf_hub_available
):
    """Test provider initializes correctly."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    # Mock model path and session
    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {"embedding_dim": 768}

    # Mock tokenizer file existence
    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file") as mock_tokenizer:
            mock_tokenizer.return_value = Mock()
            
            provider = ONNXLocalProvider(onnx_config)

            assert provider.model_name == "Xenova/all-mpnet-base-v2"
            assert provider.get_dimensions() == 768
            assert provider.get_max_batch_size() == 32


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_get_embeddings(
    mock_metadata, mock_validate, mock_download, mock_session, onnx_config, mock_onnx_available, mock_hf_hub_available
):
    """Test embedding generation."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    # Mock model path and session
    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_input = Mock()
    mock_input.name = "input_ids"
    mock_session_obj.get_inputs.return_value = [mock_input]
    
    # Mock ONNX inference
    mock_embeddings = np.random.randn(2, 512, 768).astype(np.float32)
    mock_session_obj.run.return_value = [mock_embeddings]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {"embedding_dim": 768}

    # Mock tokenizer
    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file") as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_encoding = Mock()
            mock_encoding.ids = [1, 2, 3, 4, 5]
            mock_tokenizer.encode.return_value = mock_encoding
            mock_tokenizer_class.return_value = mock_tokenizer
            
            provider = ONNXLocalProvider(onnx_config)

            # Generate embeddings
            texts = ["hello world", "test text"]
            result = provider.get_embeddings(texts)

            assert result.provider == "onnx-local"
            assert result.model == "Xenova/all-mpnet-base-v2"
            assert result.dimensions == 768
            assert len(result.embeddings) == 2
            assert result.cost == 0.0
            assert result.cached == [False, False]


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_calculate_cost_is_zero(
    mock_metadata, mock_validate, mock_download, mock_session, onnx_config, mock_onnx_available, mock_hf_hub_available
):
    """Test that local models have zero cost."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {"embedding_dim": 768}

    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file"):
            provider = ONNXLocalProvider(onnx_config)

            cost = provider.calculate_cost(1000)
            assert cost == 0.0


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_known_model_dimensions(
    mock_metadata, mock_validate, mock_download, mock_session, mock_onnx_available, mock_hf_hub_available
):
    """Test that known models use predefined dimensions."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {}  # No metadata, should use KNOWN_MODELS

    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file"):
            config = {"model": "Xenova/all-mpnet-base-v2"}
            provider = ONNXLocalProvider(config)

            assert provider.get_dimensions() == 768  # From KNOWN_MODELS


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_unknown_model_uses_output_shape(
    mock_metadata, mock_validate, mock_download, mock_session, mock_onnx_available, mock_hf_hub_available
):
    """Test that unknown models infer dimensions from output shape."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 512]  # Custom dimension
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {}  # No metadata

    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file"):
            config = {"model": "some-unknown-model"}
            provider = ONNXLocalProvider(config)

            assert provider.get_dimensions() == 512  # From output shape


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_custom_batch_size(
    mock_metadata, mock_validate, mock_download, mock_session, mock_onnx_available, mock_hf_hub_available
):
    """Test custom batch size configuration."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {"embedding_dim": 768}

    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file"):
            config = {"model": "test-model", "max_batch_size": 64}
            provider = ONNXLocalProvider(config)

            assert provider.get_max_batch_size() == 64


@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
def test_model_loading_failure(
    mock_download, onnx_config, mock_onnx_available, mock_hf_hub_available
):
    """Test that model loading failures are handled."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider

    # Make model download fail
    mock_download.side_effect = Exception("Model not found")

    with pytest.raises(Exception, match="Model not found"):
        ONNXLocalProvider(onnx_config)


@patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session")
@patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model")
@patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata")
def test_get_model_name(
    mock_metadata, mock_validate, mock_download, mock_session, onnx_config, mock_onnx_available, mock_hf_hub_available
):
    """Test getting model name."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    from pathlib import Path

    mock_download.return_value = Path("/fake/path/to/onnx/model.onnx")
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, 768]
    mock_session_obj.get_outputs.return_value = [mock_output]
    mock_session.return_value = mock_session_obj
    mock_metadata.return_value = {"embedding_dim": 768}

    with patch("pathlib.Path.exists", return_value=True):
        with patch("tokenizers.Tokenizer.from_file"):
            provider = ONNXLocalProvider(onnx_config)

            assert provider.get_model_name() == "Xenova/all-mpnet-base-v2"


def test_mean_pooling():
    """Test mean pooling implementation."""
    from rag_factory.services.utils.onnx_utils import mean_pooling
    import numpy as np

    # Create test data
    token_embeddings = np.random.randn(2, 10, 768).astype(np.float32)
    attention_mask = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int64)

    # Call mean pooling utility function
    result = mean_pooling(token_embeddings, attention_mask)

    # Check output shape
    assert result.shape == (2, 768)
    # Check that result is not all zeros
    assert not np.allclose(result, 0)
