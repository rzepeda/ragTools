"""Unit tests for base reranking classes."""

import pytest
from rag_factory.strategies.reranking.base import (
    IReranker,
    RerankerModel,
    RerankResult,
    RerankResponse,
    RerankConfig,
)


class TestRerankConfig:
    """Tests for RerankConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankConfig()

        assert config.model == RerankerModel.CROSS_ENCODER
        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.initial_retrieval_size == 100
        assert config.top_k == 10
        assert config.score_threshold == 0.0
        assert config.normalize_scores is True
        assert config.batch_size == 32
        assert config.enable_cache is True
        assert config.cache_ttl == 3600
        assert config.timeout_seconds == 5.0
        assert config.enable_fallback is True
        assert config.fallback_to_vector_scores is True
        assert config.model_config == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RerankConfig(
            model=RerankerModel.COHERE,
            model_name="rerank-english-v2.0",
            initial_retrieval_size=50,
            top_k=5,
            score_threshold=0.5,
            normalize_scores=False,
            batch_size=16,
            enable_cache=False,
            model_config={"api_key": "test-key"}
        )

        assert config.model == RerankerModel.COHERE
        assert config.model_name == "rerank-english-v2.0"
        assert config.initial_retrieval_size == 50
        assert config.top_k == 5
        assert config.score_threshold == 0.5
        assert config.normalize_scores is False
        assert config.batch_size == 16
        assert config.enable_cache is False
        assert config.model_config == {"api_key": "test-key"}


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test creating a rerank result."""
        result = RerankResult(
            document_id="doc1",
            original_rank=5,
            reranked_rank=0,
            original_score=0.7,
            rerank_score=0.95,
            normalized_score=1.0
        )

        assert result.document_id == "doc1"
        assert result.original_rank == 5
        assert result.reranked_rank == 0
        assert result.original_score == 0.7
        assert result.rerank_score == 0.95
        assert result.normalized_score == 1.0


class TestRerankResponse:
    """Tests for RerankResponse dataclass."""

    def test_rerank_response_creation(self):
        """Test creating a rerank response."""
        results = [
            RerankResult("doc1", 0, 0, 0.9, 0.95, 1.0),
            RerankResult("doc2", 1, 1, 0.8, 0.85, 0.5)
        ]

        response = RerankResponse(
            query="test query",
            results=results,
            total_candidates=10,
            reranked_count=2,
            top_k=5,
            model_used="test-model",
            execution_time_ms=150.5,
            cache_hit=False
        )

        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.total_candidates == 10
        assert response.reranked_count == 2
        assert response.top_k == 5
        assert response.model_used == "test-model"
        assert response.execution_time_ms == 150.5
        assert response.cache_hit is False
        assert response.metadata == {}

    def test_rerank_response_with_metadata(self):
        """Test creating a rerank response with metadata."""
        response = RerankResponse(
            query="test",
            results=[],
            total_candidates=0,
            reranked_count=0,
            top_k=10,
            model_used="test",
            execution_time_ms=0.0,
            cache_hit=False,
            metadata={"fallback": True}
        )

        assert response.metadata == {"fallback": True}


class MockReranker(IReranker):
    """Mock reranker for testing."""

    def rerank(self, query, documents, scores=None):
        # Return documents in reverse order with mock scores
        return [(i, 1.0 - i * 0.1) for i in range(len(documents))]

    def get_model_name(self):
        return "mock-reranker"


class TestIReranker:
    """Tests for IReranker base class."""

    def test_normalize_scores_basic(self):
        """Test score normalization."""
        config = RerankConfig()
        reranker = MockReranker(config)

        scores = [0.5, 0.75, 1.0, 0.25]
        normalized = reranker.normalize_scores(scores)

        assert len(normalized) == 4
        assert all(0.0 <= s <= 1.0 for s in normalized)
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0

    def test_normalize_scores_empty(self):
        """Test normalization with empty list."""
        config = RerankConfig()
        reranker = MockReranker(config)

        normalized = reranker.normalize_scores([])
        assert normalized == []

    def test_normalize_scores_all_same(self):
        """Test normalization when all scores are identical."""
        config = RerankConfig()
        reranker = MockReranker(config)

        scores = [0.8, 0.8, 0.8]
        normalized = reranker.normalize_scores(scores)

        assert all(s == 1.0 for s in normalized)

    def test_validate_inputs_empty_query(self):
        """Test validation rejects empty query."""
        config = RerankConfig()
        reranker = MockReranker(config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.validate_inputs("", ["doc"])

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.validate_inputs("   ", ["doc"])

    def test_validate_inputs_empty_documents(self):
        """Test validation rejects empty documents list."""
        config = RerankConfig()
        reranker = MockReranker(config)

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            reranker.validate_inputs("query", [])

    def test_validate_inputs_too_many_documents(self):
        """Test validation rejects too many documents."""
        config = RerankConfig()
        reranker = MockReranker(config)

        with pytest.raises(ValueError, match="Too many documents"):
            reranker.validate_inputs("query", ["doc"] * 501)

    def test_validate_inputs_valid(self):
        """Test validation passes with valid inputs."""
        config = RerankConfig()
        reranker = MockReranker(config)

        # Should not raise
        reranker.validate_inputs("query", ["doc1", "doc2"])

    def test_mock_reranker_rerank(self):
        """Test mock reranker implementation."""
        config = RerankConfig()
        reranker = MockReranker(config)

        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_mock_reranker_get_model_name(self):
        """Test getting model name."""
        config = RerankConfig()
        reranker = MockReranker(config)

        assert reranker.get_model_name() == "mock-reranker"
