"""Unit tests for SelfReflectiveRAGStrategy."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.strategies.self_reflective.strategy import SelfReflectiveRAGStrategy
from rag_factory.strategies.self_reflective.models import (
    Grade, GradeLevel, QueryRefinement, RefinementStrategy
)


@pytest.fixture
def mock_base_strategy():
    """Create a mock base strategy."""
    strategy = Mock()
    strategy.retrieve.return_value = [
        {"chunk_id": "c1", "text": "Result 1", "score": 0.9},
        {"chunk_id": "c2", "text": "Result 2", "score": 0.8}
    ]
    return strategy


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    return Mock()


@pytest.fixture
def self_reflective_strategy(mock_base_strategy, mock_llm_service):
    """Create a SelfReflectiveRAGStrategy instance."""
    config = {
        "grade_threshold": 4.0,
        "max_retries": 2
    }
    return SelfReflectiveRAGStrategy(
        mock_base_strategy,
        mock_llm_service,
        config
    )


def test_retrieve_good_results_no_retry(self_reflective_strategy, mock_base_strategy):
    """Test that good results don't trigger retry."""
    with patch.object(self_reflective_strategy.grader, 'grade_results') as mock_grader:
        # Mock high grades
        mock_grader.return_value = [
            Grade(chunk_id="c1", score=4.5, level=GradeLevel.EXCELLENT,
                  relevance=0.9, completeness=0.9, reasoning="Great"),
            Grade(chunk_id="c2", score=4.2, level=GradeLevel.GOOD,
                  relevance=0.85, completeness=0.8, reasoning="Good")
        ]

        results = self_reflective_strategy.retrieve("test query", top_k=5)

        # Should only call base strategy once (no retries)
        assert mock_base_strategy.retrieve.call_count == 1
        assert len(results) == 2


def test_retrieve_poor_results_triggers_retry(self_reflective_strategy, mock_base_strategy):
    """Test that poor results trigger retry."""
    with patch.object(self_reflective_strategy.grader, 'grade_results') as mock_grader, \
         patch.object(self_reflective_strategy.refiner, 'refine_query') as mock_refiner:

        # First attempt: poor grades
        # Second attempt: good grades
        mock_grader.side_effect = [
            [Grade(chunk_id="c1", score=2.0, level=GradeLevel.POOR,
                   relevance=0.3, completeness=0.3, reasoning="Poor")],
            [Grade(chunk_id="c3", score=4.5, level=GradeLevel.EXCELLENT,
                   relevance=0.9, completeness=0.9, reasoning="Excellent")]
        ]

        mock_refiner.return_value = QueryRefinement(
            original_query="test query",
            refined_query="refined test query",
            strategy=RefinementStrategy.REFORMULATION,
            reasoning="Reformulated",
            iteration=1
        )

        results = self_reflective_strategy.retrieve("test query", top_k=5)

        # Should call base strategy twice (initial + 1 retry)
        assert mock_base_strategy.retrieve.call_count == 2

        # Should have called refiner
        mock_refiner.assert_called_once()


def test_max_retries_enforced(self_reflective_strategy, mock_base_strategy):
    """Test that max retries is enforced."""
    with patch.object(self_reflective_strategy.grader, 'grade_results') as mock_grader, \
         patch.object(self_reflective_strategy.refiner, 'refine_query') as mock_refiner:

        # Always return poor grades
        mock_grader.return_value = [
            Grade(chunk_id="c1", score=2.0, level=GradeLevel.POOR,
                  relevance=0.3, completeness=0.3, reasoning="Poor")
        ]

        mock_refiner.return_value = QueryRefinement(
            original_query="test query",
            refined_query="refined query",
            strategy=RefinementStrategy.REFORMULATION,
            reasoning="Refined",
            iteration=1
        )

        results = self_reflective_strategy.retrieve("test query", top_k=5)

        # Should stop at max_retries (2) + initial = 3 total
        assert mock_base_strategy.retrieve.call_count == 3


def test_result_aggregation(self_reflective_strategy):
    """Test aggregation of results across attempts."""
    from rag_factory.strategies.self_reflective.models import RetrievalAttempt

    attempts = [
        RetrievalAttempt(
            attempt_number=1,
            query="query1",
            results=[
                {"chunk_id": "c1", "text": "Text 1", "score": 0.8},
                {"chunk_id": "c2", "text": "Text 2", "score": 0.7}
            ],
            grades=[
                Grade(chunk_id="c1", score=2.0, level=GradeLevel.POOR,
                      relevance=0.3, completeness=0.3, reasoning="Poor"),
                Grade(chunk_id="c2", score=3.0, level=GradeLevel.FAIR,
                      relevance=0.5, completeness=0.5, reasoning="Fair")
            ],
            average_grade=2.5,
            timestamp=1.0,
            latency_ms=100.0
        ),
        RetrievalAttempt(
            attempt_number=2,
            query="query2",
            results=[
                {"chunk_id": "c1", "text": "Text 1", "score": 0.85},  # Duplicate
                {"chunk_id": "c3", "text": "Text 3", "score": 0.9}
            ],
            grades=[
                Grade(chunk_id="c1", score=4.5, level=GradeLevel.EXCELLENT,
                      relevance=0.9, completeness=0.9, reasoning="Excellent"),
                Grade(chunk_id="c3", score=4.0, level=GradeLevel.GOOD,
                      relevance=0.8, completeness=0.8, reasoning="Good")
            ],
            average_grade=4.25,
            timestamp=2.0,
            latency_ms=100.0
        )
    ]

    results = self_reflective_strategy._aggregate_results(attempts, top_k=5)

    # Should have 3 unique chunks: c1 (deduplicated, higher grade), c2, c3
    assert len(results) == 3
    assert all("grade" in r for r in results)
    assert all("combined_score" in r for r in results)
    
    # c1 should use the better grade from attempt 2
    c1_result = next(r for r in results if r["chunk_id"] == "c1")
    assert c1_result["grade"] == 4.5
    assert c1_result["retrieval_attempt"] == 2

    # Results should be sorted by combined score
    assert results[0]["combined_score"] >= results[1]["combined_score"]


def test_timeout_protection(mock_base_strategy, mock_llm_service):
    """Test timeout protection."""
    config = {
        "grade_threshold": 4.0,
        "max_retries": 10,  # High retries
        "timeout_seconds": 0.1  # Very short timeout
    }
    strategy = SelfReflectiveRAGStrategy(
        mock_base_strategy,
        mock_llm_service,
        config
    )

    with patch.object(strategy.grader, 'grade_results') as mock_grader:
        # Always return poor grades
        mock_grader.return_value = [
            Grade(chunk_id="c1", score=2.0, level=GradeLevel.POOR,
                  relevance=0.3, completeness=0.3, reasoning="Poor")
        ]

        import time
        start = time.time()
        results = strategy.retrieve("test query", top_k=5)
        elapsed = time.time() - start

        # Should timeout quickly
        assert elapsed < 1.0  # Should be much less than 1 second


def test_same_query_prevention(self_reflective_strategy, mock_base_strategy):
    """Test that same query stops retry loop."""
    with patch.object(self_reflective_strategy.grader, 'grade_results') as mock_grader, \
         patch.object(self_reflective_strategy.refiner, 'refine_query') as mock_refiner:

        mock_grader.return_value = [
            Grade(chunk_id="c1", score=2.0, level=GradeLevel.POOR,
                  relevance=0.3, completeness=0.3, reasoning="Poor")
        ]

        # Refiner returns same query
        mock_refiner.return_value = QueryRefinement(
            original_query="test query",
            refined_query="test query",  # Same as original!
            strategy=RefinementStrategy.REFORMULATION,
            reasoning="Failed to refine",
            iteration=1
        )

        results = self_reflective_strategy.retrieve("test query", top_k=5)

        # Should stop after detecting same query
        assert mock_base_strategy.retrieve.call_count < 3


def test_normalize_results(self_reflective_strategy):
    """Test result normalization."""
    # Test with dict results
    dict_results = [{"chunk_id": "c1", "text": "text", "score": 0.9}]
    normalized = self_reflective_strategy._normalize_results(dict_results)
    assert normalized == dict_results

    # Test with Chunk-like objects
    class MockChunk:
        def __init__(self):
            self.chunk_id = "c1"
            self.text = "text"
            self.score = 0.9
            self.metadata = {}

    chunk_results = [MockChunk()]
    normalized = self_reflective_strategy._normalize_results(chunk_results)
    assert len(normalized) == 1
    assert normalized[0]["chunk_id"] == "c1"


def test_strategy_properties(self_reflective_strategy):
    """Test strategy properties."""
    assert self_reflective_strategy.name == "self_reflective"
    assert "self-correcting" in self_reflective_strategy.description.lower()
