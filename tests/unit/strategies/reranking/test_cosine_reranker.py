"""Unit tests for cosine similarity reranker."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from rag_factory.strategies.reranking.cosine_reranker import CosineReranker
from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel


@pytest.fixture
def mock_embedder():
    """Create a mock embedding provider."""
    embedder = Mock()
    # Query embedding: [1, 0, 0]
    embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    # Document embeddings
    embedder.embed_documents.return_value = [
        [1.0, 0.0, 0.0],  # Perfect match (cosine = 1.0)
        [0.7, 0.7, 0.0],  # Partial match (cosine ≈ 0.7)
        [0.0, 1.0, 0.0],  # Orthogonal (cosine = 0.0)
        [0.5, 0.5, 0.0],  # Another partial match
    ]
    return embedder


@pytest.fixture
def rerank_config():
    """Create a basic rerank config."""
    return RerankConfig(
        model=RerankerModel.COSINE,
        model_config={"embedding_provider": Mock()}
    )


class TestCosineRerankerInitialization:
    """Tests for CosineReranker initialization."""

    def test_initialization_success(self, mock_embedder, rerank_config):
        """Test successful initialization."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="cosine",
            normalize=True
        )

        assert reranker.embedding_provider == mock_embedder
        assert reranker.metric == "cosine"
        assert reranker.normalize is True

    def test_initialization_without_embedder(self, rerank_config):
        """Test initialization fails without embedding provider."""
        with pytest.raises(ValueError, match="requires an embedding_provider"):
            CosineReranker(rerank_config, embedding_provider=None)

    def test_initialization_invalid_metric(self, mock_embedder, rerank_config):
        """Test initialization fails with invalid metric."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            CosineReranker(
                rerank_config,
                embedding_provider=mock_embedder,
                metric="invalid_metric"
            )

    def test_supported_metrics(self, mock_embedder, rerank_config):
        """Test all supported metrics can be initialized."""
        for metric in ["cosine", "dot", "euclidean"]:
            reranker = CosineReranker(
                rerank_config,
                embedding_provider=mock_embedder,
                metric=metric
            )
            assert reranker.metric == metric


class TestCosineRerankerReranking:
    """Tests for reranking functionality."""

    def test_rerank_cosine_similarity(self, mock_embedder, rerank_config):
        """Test basic cosine similarity reranking."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="cosine",
            normalize=True
        )

        documents = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        results = reranker.rerank("query", documents)

        assert len(results) == 4
        # First result should be perfect match (index 0)
        assert results[0][0] == 0
        assert results[0][1] > 0.99  # Nearly 1.0
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_dot_product(self, mock_embedder, rerank_config):
        """Test dot product similarity reranking."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="dot",
            normalize=False
        )

        documents = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        results = reranker.rerank("query", documents)

        assert len(results) == 4
        # Results should be sorted by dot product
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_euclidean_similarity(self, mock_embedder, rerank_config):
        """Test euclidean similarity reranking."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="euclidean",
            normalize=True
        )

        documents = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        results = reranker.rerank("query", documents)

        assert len(results) == 4
        # Perfect match should have highest similarity (smallest distance)
        assert results[0][0] == 0
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_documents(self, mock_embedder, rerank_config):
        """Test reranking with empty document list."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            reranker.rerank("query", [])

    def test_rerank_empty_query(self, mock_embedder, rerank_config):
        """Test reranking with empty query."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            reranker.rerank("", ["Doc 1"])

    def test_rerank_with_normalization(self, rerank_config):
        """Test that normalization affects results correctly."""
        embedder = Mock()
        embedder.embed_query.return_value = [3.0, 4.0, 0.0]  # Length 5
        embedder.embed_documents.return_value = [
            [6.0, 8.0, 0.0],  # Same direction, length 10
            [0.0, 5.0, 0.0],  # Different direction
        ]

        reranker = CosineReranker(
            rerank_config,
            embedding_provider=embedder,
            metric="cosine",
            normalize=True
        )

        results = reranker.rerank("query", ["Doc 1", "Doc 2"])

        # With normalization, first doc should have cosine = 1.0
        assert results[0][0] == 0
        assert abs(results[0][1] - 1.0) < 0.01


class TestCosineRerankerNormalization:
    """Tests for vector normalization."""

    def test_normalize_vector(self, mock_embedder, rerank_config):
        """Test single vector normalization."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder
        )

        vector = np.array([3.0, 4.0])  # Length 5
        normalized = reranker._normalize_vector(vector)

        # Should be unit length
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6
        # Should maintain direction
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6

    def test_normalize_zero_vector(self, mock_embedder, rerank_config):
        """Test normalization of zero vector."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder
        )

        vector = np.array([0.0, 0.0])
        normalized = reranker._normalize_vector(vector)

        # Should return original vector
        np.testing.assert_array_equal(normalized, vector)

    def test_normalize_batch(self, mock_embedder, rerank_config):
        """Test batch vector normalization."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder
        )

        vectors = np.array([
            [3.0, 4.0],
            [5.0, 12.0],
            [0.0, 0.0]
        ])
        normalized = reranker._normalize_batch(vectors)

        # All should be unit length (except zero vector)
        norms = np.linalg.norm(normalized, axis=1)
        assert abs(norms[0] - 1.0) < 1e-6
        assert abs(norms[1] - 1.0) < 1e-6
        # Zero vector should remain zero
        assert norms[2] < 1e-6


class TestCosineRerankerModelName:
    """Tests for model name."""

    def test_get_model_name_cosine(self, mock_embedder, rerank_config):
        """Test model name for cosine metric."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="cosine"
        )

        assert reranker.get_model_name() == "cosine-cosine"

    def test_get_model_name_dot(self, mock_embedder, rerank_config):
        """Test model name for dot product metric."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="dot"
        )

        assert reranker.get_model_name() == "cosine-dot"

    def test_get_model_name_euclidean(self, mock_embedder, rerank_config):
        """Test model name for euclidean metric."""
        reranker = CosineReranker(
            rerank_config,
            embedding_provider=mock_embedder,
            metric="euclidean"
        )

        assert reranker.get_model_name() == "cosine-euclidean"


class TestCosineRerankerPerformance:
    """Tests for performance characteristics."""

    def test_large_batch_processing(self, rerank_config):
        """Test processing large batch of documents."""
        embedder = Mock()
        embedder.embed_query.return_value = [1.0] + [0.0] * 383  # 384-dim
        embedder.embed_documents.return_value = [
            [float(i == j) for j in range(384)]
            for i in range(100)
        ]

        reranker = CosineReranker(
            rerank_config,
            embedding_provider=embedder,
            metric="cosine"
        )

        documents = [f"Doc {i}" for i in range(100)]
        results = reranker.rerank("query", documents)

        assert len(results) == 100
        # All documents should be ranked
        indices = [idx for idx, _ in results]
        assert len(set(indices)) == 100
