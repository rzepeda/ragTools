"""Unit tests for reranker service."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from rag_factory.strategies.reranking.reranker_service import (
    RerankerService,
    CandidateDocument,
)
from rag_factory.strategies.reranking.base import (
    RerankConfig,
    RerankerModel,
    RerankResponse,
)


@pytest.fixture
def rerank_config():
    """Create a rerank config."""
    return RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        initial_retrieval_size=100,
        top_k=10,
        enable_cache=True
    )


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    reranker = Mock()
    reranker.rerank.return_value = [
        (2, 0.95),  # doc 2 has highest score
        (0, 0.85),
        (1, 0.75)
    ]
    reranker.get_model_name.return_value = "mock-reranker"
    reranker.normalize_scores.return_value = [1.0, 0.5, 0.0]
    return reranker


@pytest.fixture
def sample_candidates():
    """Create sample candidate documents."""
    return [
        CandidateDocument(id="doc1", text="Document 1", original_score=0.9),
        CandidateDocument(id="doc2", text="Document 2", original_score=0.8),
        CandidateDocument(id="doc3", text="Document 3", original_score=0.7)
    ]


class TestCandidateDocument:
    """Tests for CandidateDocument dataclass."""

    def test_candidate_creation(self):
        """Test creating a candidate document."""
        doc = CandidateDocument(
            id="doc1",
            text="Sample text",
            original_score=0.85,
            metadata={"source": "test"}
        )

        assert doc.id == "doc1"
        assert doc.text == "Sample text"
        assert doc.original_score == 0.85
        assert doc.metadata == {"source": "test"}

    def test_candidate_default_metadata(self):
        """Test default metadata is empty dict."""
        doc = CandidateDocument(id="doc1", text="text", original_score=0.9)

        assert doc.metadata == {}


class TestRerankerService:
    """Tests for RerankerService."""

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_service_initialization(self, mock_reranker_class, rerank_config):
        """Test service initializes correctly."""
        mock_reranker_class.return_value = Mock()

        service = RerankerService(rerank_config)

        assert service.config == rerank_config
        assert service.reranker is not None
        assert service.cache is not None
        mock_reranker_class.assert_called_once()

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_service_without_cache(self, mock_reranker_class):
        """Test service without cache."""
        mock_reranker_class.return_value = Mock()

        config = RerankConfig(enable_cache=False)
        service = RerankerService(config)

        assert service.cache is None

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_rerank_basic(self, mock_reranker_class, rerank_config, sample_candidates):
        """Test basic re-ranking."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (2, 0.95),
            (0, 0.85),
            (1, 0.75)
        ]
        mock_reranker.get_model_name.return_value = "mock-reranker"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)
        response = service.rerank("test query", sample_candidates)

        assert response.total_candidates == 3
        assert response.reranked_count == 3
        assert len(response.results) == 3

        # Check that doc3 is now ranked first (index 2 had highest score)
        assert response.results[0].document_id == "doc3"
        assert response.results[0].reranked_rank == 0

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_rerank_empty_candidates(self, mock_reranker_class, rerank_config):
        """Test re-ranking with empty candidates list."""
        mock_reranker_class.return_value = Mock()

        service = RerankerService(rerank_config)
        response = service.rerank("test query", [])

        assert response.total_candidates == 0
        assert response.reranked_count == 0
        assert len(response.results) == 0

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_rerank_top_k_limit(self, mock_reranker_class, sample_candidates):
        """Test that top_k limits results."""
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            top_k=2  # Only return top 2
        )

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (2, 0.95),
            (0, 0.85),
            (1, 0.75)
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(config)
        response = service.rerank("test query", sample_candidates)

        assert response.total_candidates == 3
        assert response.reranked_count == 2
        assert len(response.results) == 2

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_cache_hit(self, mock_reranker_class, rerank_config, sample_candidates):
        """Test that cache returns cached results."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)

        # First call - cache miss
        response1 = service.rerank("test query", sample_candidates[:1])
        assert response1.cache_hit is False

        # Second call - should hit cache
        response2 = service.rerank("test query", sample_candidates[:1])
        assert response2.cache_hit is True

        # Reranker should only be called once
        assert mock_reranker.rerank.call_count == 1

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_cache_disabled(self, mock_reranker_class, sample_candidates):
        """Test service with cache disabled."""
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            enable_cache=False
        )

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(config)

        service.rerank("query", sample_candidates[:1])
        service.rerank("query", sample_candidates[:1])

        # Should call reranker twice (no caching)
        assert mock_reranker.rerank.call_count == 2

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_score_threshold(self, mock_reranker_class):
        """Test score threshold filtering."""
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            score_threshold=0.8  # Filter scores below 0.8
        )

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (0, 0.95),  # Above threshold
            (1, 0.85),  # Above threshold
            (2, 0.75)   # Below threshold - should be filtered
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id=f"doc{i}", text=f"Doc {i}", original_score=0.9)
            for i in range(3)
        ]

        response = service.rerank("query", candidates)

        # Should only return 2 results (above threshold)
        assert response.reranked_count == 2

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_fallback_on_error(self, mock_reranker_class, rerank_config):
        """Test fallback to vector scores on error."""
        mock_reranker = Mock()
        mock_reranker.rerank.side_effect = Exception("Reranker failed")
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9),
            CandidateDocument(id="doc2", text="Doc 2", original_score=0.8),
            CandidateDocument(id="doc3", text="Doc 3", original_score=0.7)
        ]

        response = service.rerank("query", candidates)

        # Should fallback successfully
        assert response.model_used == "fallback_vector_similarity"
        assert len(response.results) > 0
        assert response.metadata.get("fallback") is True

        # Results should be sorted by original scores
        assert response.results[0].document_id == "doc1"
        assert response.results[1].document_id == "doc2"

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_no_fallback_raises_error(self, mock_reranker_class):
        """Test that errors are raised when fallback is disabled."""
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            enable_fallback=False
        )

        mock_reranker = Mock()
        mock_reranker.rerank.side_effect = Exception("Reranker failed")
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9)
        ]

        with pytest.raises(Exception, match="Reranker failed"):
            service.rerank("query", candidates)

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_get_stats(self, mock_reranker_class, rerank_config, sample_candidates):
        """Test statistics tracking."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)

        service.rerank("query1", sample_candidates[:1])
        service.rerank("query1", sample_candidates[:1])  # Cache hit
        service.rerank("query2", sample_candidates[:1])  # Cache miss

        stats = service.get_stats()

        assert stats["total_requests"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(0.333, 0.01)
        assert stats["model"] == "mock"

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_clear_cache(self, mock_reranker_class, rerank_config, sample_candidates):
        """Test clearing cache."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)

        # Populate cache
        service.rerank("query", sample_candidates[:1])

        # Clear cache
        service.clear_cache()

        # Next call should be a cache miss
        service.rerank("query", sample_candidates[:1])

        stats = service.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 2

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_score_normalization(self, mock_reranker_class, rerank_config, sample_candidates):
        """Test score normalization."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (0, 1.0),
            (1, 0.5),
            (2, 0.0)
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.return_value = [1.0, 0.5, 0.0]
        mock_reranker_class.return_value = mock_reranker

        service = RerankerService(rerank_config)
        response = service.rerank("query", sample_candidates)

        # Check normalized scores
        assert response.results[0].normalized_score == 1.0
        assert response.results[1].normalized_score == 0.5
        assert response.results[2].normalized_score == 0.0

    @patch('rag_factory.strategies.reranking.reranker_service.CohereReranker')
    def test_cohere_model_initialization(self, mock_cohere_class):
        """Test initialization with Cohere model."""
        config = RerankConfig(
            model=RerankerModel.COHERE,
            model_config={"api_key": "test-key"}
        )

        mock_cohere_class.return_value = Mock()

        service = RerankerService(config)

        mock_cohere_class.assert_called_once_with(config)
