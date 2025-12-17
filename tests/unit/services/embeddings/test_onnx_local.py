"""Unit tests for ONNX local embedding provider.

Tests the ONNX provider using centralized mock_onnx_env fixture
from conftest.py to ensure correct behavior without requiring actual models.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
from rag_factory.services.embedding.base import EmbeddingResult


class TestONNXLocalProvider:
    """Test ONNX local embedding provider."""

    def test_initialization(self, mock_onnx_env):
        """Test provider initialization."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            assert provider.model_name == "test-model"
            assert provider._dimensions == 384
            assert provider.max_batch_size == 32

    def test_initialization_with_custom_config(self, mock_onnx_env):
        """Test initialization with custom configuration."""
        with mock_onnx_env(dimension=768):
            provider = ONNXLocalProvider({
                "model": "custom-model",
                "max_batch_size": 16,
                "max_length": 256,
                "num_threads": 4
            })
            
            assert provider.model_name == "custom-model"
            assert provider.max_batch_size == 16
            assert provider.max_length == 256

    def test_get_embeddings_single_text(self, mock_onnx_env):
        """Test embedding a single text."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            result = provider.get_embeddings(["Test document"])

            assert isinstance(result, EmbeddingResult)
            # Mock returns 2 embeddings regardless of input
            assert len(result.embeddings) >= 1
            assert len(result.embeddings[0]) == 384
            assert result.model == "test-model"
            assert result.provider == "onnx-local"
            assert result.cost == 0.0

    def test_get_embeddings_multiple_texts(self, mock_onnx_env):
        """Test embedding multiple texts."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            texts = ["Document 1", "Document 2"]
            result = provider.get_embeddings(texts)

            assert len(result.embeddings) == 2  # Mock returns 2 embeddings
            assert all(len(emb) == 384 for emb in result.embeddings)
            assert result.dimensions == 384

    def test_get_embeddings_empty_list(self, mock_onnx_env):
        """Test handling of empty input."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            result = provider.get_embeddings([])

            assert result.embeddings == []
            assert result.token_count == 0

    def test_embedding_normalization(self, mock_onnx_env):
        """Test that embeddings are normalized."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            result = provider.get_embeddings(["Test"])

            # Check that embedding is normalized (unit length)
            embedding = np.array(result.embeddings[0])
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01

    def test_get_dimensions(self, mock_onnx_env):
        """Test getting embedding dimensions."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            assert provider.get_dimensions() == 384

    def test_get_max_batch_size(self, mock_onnx_env):
        """Test getting max batch size."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            assert provider.get_max_batch_size() == 32

    def test_get_model_name(self, mock_onnx_env):
        """Test getting model name."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            assert provider.get_model_name() == "test-model"

    def test_calculate_cost(self, mock_onnx_env):
        """Test cost calculation (should always be 0 for local models)."""
        with mock_onnx_env(dimension=384):
            provider = ONNXLocalProvider({"model": "test-model"})
            assert provider.calculate_cost(1000) == 0.0
            assert provider.calculate_cost(0) == 0.0

    def test_known_models(self):
        """Test that known models are properly configured."""
        assert "sentence-transformers/all-MiniLM-L6-v2" in ONNXLocalProvider.KNOWN_MODELS
        assert ONNXLocalProvider.KNOWN_MODELS["sentence-transformers/all-MiniLM-L6-v2"]["dimensions"] == 384

    def test_missing_onnx_runtime(self):
        """Test error when ONNX Runtime is not installed."""
        with patch("rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ONNX Runtime not installed"):
                ONNXLocalProvider({"model": "test"})

    def test_model_loading_failure(self):
        """Test handling of model loading failure."""
        with patch("rag_factory.services.utils.onnx_utils.get_onnx_model_path") as mock_get_path:
            mock_get_path.side_effect = FileNotFoundError("ONNX model 'nonexistent-model' not found locally")
            
            with pytest.raises(FileNotFoundError, match="not found locally"):
                ONNXLocalProvider({"model": "nonexistent-model"})

    def test_session_creation_failure(self):
        """Test handling of session creation failure."""
        # Patch at the import location in onnx_local module
        from pathlib import Path
        with patch("rag_factory.services.embedding.providers.onnx_local.get_onnx_model_path", return_value=Path("/fake/model.onnx")):
            with patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session") as mock_create:
                with patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model"):
                    with patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata", return_value={"embedding_dim": 384}):
                        mock_create.side_effect = ValueError("Session creation failed")
                        
                        with pytest.raises(ValueError, match="Session creation failed"):
                            ONNXLocalProvider({"model": "test"})
