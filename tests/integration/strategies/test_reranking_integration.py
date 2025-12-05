"""Integration tests for reranking strategies."""

import pytest
from unittest.mock import Mock, patch

from rag_factory.strategies.reranking.reranker_service import (
    RerankerService,
    CandidateDocument,
)
from rag_factory.strategies.reranking.base import (
    RerankConfig,
    RerankerModel,
)


@pytest.fixture
def sample_candidates_large():
    """Create a larger set of candidate documents for testing."""
    candidates = []

    # Highly relevant documents
    candidates.extend([
        CandidateDocument(
            id="relevant1",
            text="Machine learning algorithms include decision trees, neural networks, and support vector machines.",
            original_score=0.6
        ),
        CandidateDocument(
            id="relevant2",
            text="Supervised learning algorithms learn from labeled data.",
            original_score=0.7
        ),
        CandidateDocument(
            id="relevant3",
            text="Deep learning is a subset of machine learning.",
            original_score=0.65
        ),
    ])

    # Irrelevant documents with higher original scores
    candidates.extend([
        CandidateDocument(
            id="irrelevant1",
            text="The weather today is sunny and warm.",
            original_score=0.9
        ),
        CandidateDocument(
            id="irrelevant2",
            text="Cooking recipes for beginners.",
            original_score=0.8
        ),
    ])

    return candidates


class TestMockRerankerIntegration:
    """Integration tests using mock reranker (no external dependencies)."""

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_end_to_end_reranking(self, mock_reranker_class, sample_candidates_large):
        """Test complete re-ranking flow."""
        # Setup mock to simulate reranking improvement
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (0, 0.95),  # relevant1
            (1, 0.90),  # relevant2
            (2, 0.85),  # relevant3
            (3, 0.30),  # irrelevant1
            (4, 0.25),  # irrelevant2
        ]
        mock_reranker.get_model_name.return_value = "mock-cross-encoder"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            top_k=3
        )

        service = RerankerService(config)
        response = service.rerank("machine learning algorithms", sample_candidates_large)

        # Verify response structure
        assert response.query == "machine learning algorithms"
        assert response.total_candidates == 5
        assert response.reranked_count == 3
        assert len(response.results) == 3

        # Verify relevant documents are ranked higher
        top_3_ids = [r.document_id for r in response.results]
        assert "relevant1" in top_3_ids
        assert "relevant2" in top_3_ids
        assert "relevant3" in top_3_ids

        # Verify metadata
        assert response.model_used == "mock-cross-encoder"
        assert response.execution_time_ms > 0

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_performance_with_many_candidates(self, mock_reranker_class):
        """Test re-ranking performance with large candidate set."""
        # Create 100 candidates
        candidates = [
            CandidateDocument(
                id=f"doc{i}",
                text=f"Document {i} about various topics.",
                original_score=0.9 - (i * 0.001)
            )
            for i in range(100)
        ]

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (i, 1.0 - i * 0.01) for i in range(100)
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            top_k=10
        )

        service = RerankerService(config)
        response = service.rerank("test query", candidates)

        # Should handle 100 candidates efficiently
        assert response.total_candidates == 100
        assert response.reranked_count == 10
        assert response.execution_time_ms < 5000  # Should be fast with mock

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_cache_effectiveness(self, mock_reranker_class):
        """Test cache improves performance on repeated queries."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            enable_cache=True
        )

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9),
            CandidateDocument(id="doc2", text="Doc 2", original_score=0.8),
        ]

        # First call
        response1 = service.rerank("query", candidates)
        time1 = response1.execution_time_ms

        # Second call (should hit cache)
        response2 = service.rerank("query", candidates)

        assert response2.cache_hit is True
        assert mock_reranker.rerank.call_count == 1

        stats = service.get_stats()
        assert stats["cache_hit_rate"] == 0.5

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_score_threshold_filtering(self, mock_reranker_class):
        """Test score threshold filters low-relevance results."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (0, 0.95),
            (1, 0.85),
            (2, 0.75),
            (3, 0.45),
            (4, 0.35),
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            score_threshold=0.7,  # Filter below 0.7
            top_k=10
        )

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id=f"doc{i}", text=f"Doc {i}", original_score=0.9)
            for i in range(5)
        ]

        response = service.rerank("query", candidates)

        # Only 3 documents should pass threshold
        assert response.reranked_count == 3
        assert all(r.rerank_score >= 0.7 for r in response.results)

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_ranking_position_tracking(self, mock_reranker_class):
        """Test tracking of ranking position changes."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (2, 0.95),  # Was 3rd, now 1st
            (0, 0.85),  # Was 1st, now 2nd
            (1, 0.75),  # Was 2nd, now 3rd
        ]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            top_k=3
        )

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9),
            CandidateDocument(id="doc2", text="Doc 2", original_score=0.8),
            CandidateDocument(id="doc3", text="Doc 3", original_score=0.7),
        ]

        response = service.rerank("query", candidates)

        # Verify position changes
        assert response.results[0].original_rank == 2
        assert response.results[0].reranked_rank == 0

        assert response.results[1].original_rank == 0
        assert response.results[1].reranked_rank == 1

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_fallback_preserves_order(self, mock_reranker_class):
        """Test fallback maintains original score order."""
        mock_reranker = Mock()
        mock_reranker.rerank.side_effect = Exception("Model failed")
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            enable_fallback=True,
            top_k=3
        )

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9),
            CandidateDocument(id="doc2", text="Doc 2", original_score=0.95),  # Highest
            CandidateDocument(id="doc3", text="Doc 3", original_score=0.7),
        ]

        response = service.rerank("query", candidates)

        # Should fallback and preserve original score order
        assert response.metadata.get("fallback") is True
        assert response.results[0].document_id == "doc2"  # Highest original score
        assert response.results[1].document_id == "doc1"

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_batch_processing(self, mock_reranker_class):
        """Test batch processing with different batch sizes."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(i, 0.9 - i * 0.1) for i in range(10)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            batch_size=3,  # Small batch size
            top_k=5
        )

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id=f"doc{i}", text=f"Doc {i}", original_score=0.9)
            for i in range(10)
        ]

        response = service.rerank("query", candidates)

        assert response.total_candidates == 10
        assert response.reranked_count == 5

    @patch('rag_factory.strategies.reranking.reranker_service.CrossEncoderReranker')
    def test_statistics_accumulation(self, mock_reranker_class):
        """Test that statistics accumulate correctly over multiple calls."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [(0, 0.9)]
        mock_reranker.get_model_name.return_value = "mock"
        mock_reranker.normalize_scores.side_effect = lambda x: x
        mock_reranker_class.return_value = mock_reranker

        config = RerankConfig(model=RerankerModel.CROSS_ENCODER)

        service = RerankerService(config)

        candidates = [
            CandidateDocument(id="doc1", text="Doc 1", original_score=0.9)
        ]

        # Make multiple calls
        for i in range(5):
            service.rerank(f"query{i}", candidates)

        stats = service.get_stats()

        assert stats["total_requests"] == 5
        assert stats["total_documents_reranked"] == 5
        assert stats["avg_execution_time_ms"] > 0


@pytest.mark.skipif(
    True,  # Skip by default as it requires actual model
    reason="Requires sentence-transformers installation"
)
class TestRealCrossEncoderIntegration:
    """Integration tests with real cross-encoder model (optional)."""

    def test_real_model_reranking(self):
        """Test with actual cross-encoder model."""
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            model_name="ms-marco-MiniLM-L-6-v2",
            top_k=2
        )

        service = RerankerService(config)

        query = "What is the capital of France?"

        candidates = [
            CandidateDocument(
                id="doc1",
                text="Paris is the capital and largest city of France.",
                original_score=0.7
            ),
            CandidateDocument(
                id="doc2",
                text="London is the capital of the United Kingdom.",
                original_score=0.9  # Higher vector score but less relevant
            ),
        ]

        response = service.rerank(query, candidates)

        # doc1 should be ranked first despite lower original score
        assert response.results[0].document_id == "doc1"
        assert response.execution_time_ms < 2000  # Performance requirement
