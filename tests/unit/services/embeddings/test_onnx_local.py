"""Unit tests for ONNX local embedding provider.

Tests the ONNX provider with mocked models to ensure correct behavior
without requiring actual model downloads.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
from rag_factory.services.embedding.base import EmbeddingResult


class TestONNXLocalProvider:
    """Test ONNX local embedding provider."""

    @pytest.fixture
    def mock_session(self):
        """Mock ONNX Runtime session."""
        session = Mock()
        
        # Mock outputs
        output_meta = Mock()
        output_meta.shape = [1, 512, 384]  # [batch, seq_len, embedding_dim]
        output_meta.name = "last_hidden_state"
        
        session.get_outputs.return_value = [output_meta]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        
        # Mock inputs
        input_meta = Mock()
        input_meta.name = "input_ids"
        input_meta.shape = [1, 512]
        
        attention_meta = Mock()
        attention_meta.name = "attention_mask"
        attention_meta.shape = [1, 512]
        
        session.get_inputs.return_value = [input_meta, attention_meta]
        
        return session

    @pytest.fixture
    def mock_provider(self, mock_session):
        """Create embedding provider with mocked session."""
        with patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model") as mock_download:
            mock_download.return_value = Path("/fake/model.onnx")
            
            with patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session") as mock_create:
                mock_create.return_value = mock_session
                
                with patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model"):
                    with patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata") as mock_meta:
                        mock_meta.return_value = {
                            "input_names": ["input_ids", "attention_mask"],
                            "output_names": ["last_hidden_state"],
                            "embedding_dim": 384
                        }
                        
                        provider = ONNXLocalProvider({"model": "test-model"})
                        provider.session = mock_session
                        
                        return provider

    def test_initialization(self, mock_provider):
        """Test provider initialization."""
        assert mock_provider.model_name == "test-model"
        assert mock_provider._dimensions == 384
        assert mock_provider.max_batch_size == 32

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        with patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model"):
            with patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session"):
                with patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model"):
                    with patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata") as mock_meta:
                        mock_meta.return_value = {"embedding_dim": 768}
                        
                        provider = ONNXLocalProvider({
                            "model": "custom-model",
                            "max_batch_size": 16,
                            "max_length": 256,
                            "num_threads": 4
                        })
                        
                        assert provider.model_name == "custom-model"
                        assert provider.max_batch_size == 16
                        assert provider.max_length == 256

    def test_get_embeddings_single_text(self, mock_provider, mock_session):
        """Test embedding a single text."""
        # Mock session output
        mock_embeddings = np.random.randn(1, 512, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        result = mock_provider.get_embeddings(["Test document"])

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 384
        assert result.model == "test-model"
        assert result.provider == "onnx-local"
        assert result.cost == 0.0

    def test_get_embeddings_multiple_texts(self, mock_provider, mock_session):
        """Test embedding multiple texts."""
        texts = ["Document 1", "Document 2", "Document 3"]
        
        # Mock session output
        mock_embeddings = np.random.randn(3, 512, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        result = mock_provider.get_embeddings(texts)

        assert len(result.embeddings) == 3
        assert all(len(emb) == 384 for emb in result.embeddings)
        assert result.dimensions == 384

    def test_get_embeddings_empty_list(self, mock_provider):
        """Test handling of empty input."""
        result = mock_provider.get_embeddings([])

        assert result.embeddings == []
        assert result.token_count == 0

    def test_embedding_normalization(self, mock_provider, mock_session):
        """Test that embeddings are normalized."""
        # Create unnormalized embeddings
        mock_embeddings = np.array([[[3.0, 4.0] + [0.0] * 382] * 512]).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        result = mock_provider.get_embeddings(["Test"])

        # Check that embedding is normalized (unit length)
        embedding = np.array(result.embeddings[0])
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_tokenization_with_tiktoken(self, mock_provider, mock_session):
        """Test tokenization with tiktoken."""
        if mock_provider.tokenizer_type == "tiktoken":
            mock_embeddings = np.random.randn(1, 512, 384).astype(np.float32)
            mock_session.run.return_value = [mock_embeddings]

            result = mock_provider.get_embeddings(["Test document with tiktoken"])

            # Verify session was called with correct inputs
            assert mock_session.run.called
            call_args = mock_session.run.call_args
            assert "input_ids" in call_args[0][1]
            assert "attention_mask" in call_args[0][1]

    def test_tokenization_simple_fallback(self, mock_provider, mock_session):
        """Test simple tokenization fallback."""
        # Force simple tokenization
        mock_provider.tokenizer_type = "simple"
        mock_provider.tokenizer = None

        mock_embeddings = np.random.randn(1, 512, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        result = mock_provider.get_embeddings(["Test document"])

        assert len(result.embeddings) == 1

    def test_get_dimensions(self, mock_provider):
        """Test getting embedding dimensions."""
        assert mock_provider.get_dimensions() == 384

    def test_get_max_batch_size(self, mock_provider):
        """Test getting max batch size."""
        assert mock_provider.get_max_batch_size() == 32

    def test_get_model_name(self, mock_provider):
        """Test getting model name."""
        assert mock_provider.get_model_name() == "test-model"

    def test_calculate_cost(self, mock_provider):
        """Test cost calculation (should always be 0 for local models)."""
        assert mock_provider.calculate_cost(1000) == 0.0
        assert mock_provider.calculate_cost(0) == 0.0

    def test_known_models(self):
        """Test that known models are properly configured."""
        assert "sentence-transformers/all-MiniLM-L6-v2" in ONNXLocalProvider.KNOWN_MODELS
        assert ONNXLocalProvider.KNOWN_MODELS["sentence-transformers/all-MiniLM-L6-v2"]["dimensions"] == 384

    def test_missing_onnx_runtime(self):
        """Test error when ONNX Runtime is not installed."""
        with patch("rag_factory.services.embedding.providers.onnx_local.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ONNX Runtime not installed"):
                ONNXLocalProvider({"model": "test"})

    def test_missing_huggingface_hub(self):
        """Test error when HuggingFace Hub is not installed."""
        with patch("rag_factory.services.embedding.providers.onnx_local.HF_HUB_AVAILABLE", False):
            with pytest.raises(ImportError, match="HuggingFace Hub not installed"):
                ONNXLocalProvider({"model": "test"})

    def test_model_download_failure(self):
        """Test handling of model download failure."""
        with patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model") as mock_download:
            mock_download.side_effect = ValueError("Download failed")
            
            with pytest.raises(ValueError, match="Download failed"):
                ONNXLocalProvider({"model": "nonexistent-model"})

    def test_session_creation_failure(self):
        """Test handling of session creation failure."""
        with patch("rag_factory.services.embedding.providers.onnx_local.download_onnx_model"):
            with patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session") as mock_create:
                mock_create.side_effect = ValueError("Session creation failed")
                
                with pytest.raises(ValueError, match="Session creation failed"):
                    ONNXLocalProvider({"model": "test"})

    def test_embedding_error_handling(self, mock_provider, mock_session):
        """Test error handling during embedding generation."""
        mock_session.run.side_effect = Exception("Inference failed")

        with pytest.raises(Exception, match="Inference failed"):
            mock_provider.get_embeddings(["Test"])

    def test_batch_processing(self, mock_provider, mock_session):
        """Test batch processing of multiple documents."""
        # Create a batch of 10 documents
        texts = [f"Document {i}" for i in range(10)]
        
        mock_embeddings = np.random.randn(10, 512, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        result = mock_provider.get_embeddings(texts)

        assert len(result.embeddings) == 10
        assert all(isinstance(emb, list) for emb in result.embeddings)
        assert all(len(emb) == 384 for emb in result.embeddings)
