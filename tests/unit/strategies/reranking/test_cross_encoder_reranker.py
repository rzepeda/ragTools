"""Unit tests for cross-encoder reranker."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel
from rag_factory.strategies.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


@pytest.fixture
def rerank_config():
    """Create a rerank config."""
    return RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        model_name="ms-marco-MiniLM-L-6-v2",
        batch_size=32
    )


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed"
)
class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_initialization(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test reranker initializes correctly."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_config)

        assert reranker.config == rerank_config
        assert reranker.model == mock_model
        mock_cross_encoder_class.assert_called_once()

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_initialization_with_cuda(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test initialization with CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_model = Mock()
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_config)

        # Should use CUDA device
        call_args = mock_cross_encoder_class.call_args
        assert call_args is not None

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_rerank_basic(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test basic re-ranking."""
        mock_torch.cuda.is_available.return_value = False

        # Mock the model
        mock_model = Mock()
        mock_scores = Mock()
        mock_scores.tolist.return_value = [0.95, 0.75, 0.85]
        mock_model.predict.return_value = mock_scores
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_config)

        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of AI",
            "Python is a programming language",
            "ML uses algorithms to learn from data"
        ]

        results = reranker.rerank(query, documents)

        # Should return sorted results
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # First result should have highest score
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]

        # Model predict should be called
        mock_model.predict.assert_called_once()

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_rerank_batching(self, mock_torch, mock_cross_encoder_class):
        """Test batching with custom batch size."""
        mock_torch.cuda.is_available.return_value = False

        config = RerankConfig(batch_size=2)

        mock_model = Mock()
        mock_scores1 = Mock()
        mock_scores1.tolist.return_value = [0.9, 0.8]
        mock_scores2 = Mock()
        mock_scores2.tolist.return_value = [0.7]

        mock_model.predict.side_effect = [mock_scores1, mock_scores2]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(config)

        documents = ["doc1", "doc2", "doc3"]
        results = reranker.rerank("query", documents)

        # Should call predict twice (batch_size=2, 3 documents)
        assert mock_model.predict.call_count == 2
        assert len(results) == 3

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_validate_empty_query(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test validation rejects empty query."""
        mock_torch.cuda.is_available.return_value = False
        mock_cross_encoder_class.return_value = Mock()

        reranker = CrossEncoderReranker(rerank_config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", ["doc"])

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_validate_empty_documents(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test validation rejects empty documents."""
        mock_torch.cuda.is_available.return_value = False
        mock_cross_encoder_class.return_value = Mock()

        reranker = CrossEncoderReranker(rerank_config)

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            reranker.rerank("query", [])

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_validate_too_many_documents(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test validation rejects too many documents."""
        mock_torch.cuda.is_available.return_value = False
        mock_cross_encoder_class.return_value = Mock()

        reranker = CrossEncoderReranker(rerank_config)

        with pytest.raises(ValueError, match="Too many documents"):
            reranker.rerank("query", ["doc"] * 501)

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_get_model_name(self, mock_torch, mock_cross_encoder_class, rerank_config):
        """Test getting model name."""
        mock_torch.cuda.is_available.return_value = False
        mock_cross_encoder_class.return_value = Mock()

        reranker = CrossEncoderReranker(rerank_config)

        model_name = reranker.get_model_name()
        assert "marco" in model_name.lower()

    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
    @patch('rag_factory.strategies.reranking.cross_encoder_reranker.torch')
    def test_model_name_shortcut(self, mock_torch, mock_cross_encoder_class):
        """Test model name shortcuts."""
        mock_torch.cuda.is_available.return_value = False
        mock_cross_encoder_class.return_value = Mock()

        config = RerankConfig(model_name="ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(config)

        # Should expand to full model name
        assert reranker.model_name.startswith("cross-encoder/")


def test_import_error_without_dependencies():
    """Test that ImportError is raised when dependencies are missing."""
    with patch('rag_factory.strategies.reranking.cross_encoder_reranker.SENTENCE_TRANSFORMERS_AVAILABLE', False):
        config = RerankConfig()

        with pytest.raises(ImportError, match="sentence-transformers and torch are required"):
            CrossEncoderReranker(config)
