"""Unit tests for ResultDeduplicator."""

import pytest
from unittest.mock import Mock
from rag_factory.strategies.multi_query.deduplicator import ResultDeduplicator
from rag_factory.strategies.multi_query.config import MultiQueryConfig


@pytest.fixture
def config():
    """Create default config."""
    return MultiQueryConfig()


@pytest.fixture
def deduplicator(config):
    """Create deduplicator instance."""
    return ResultDeduplicator(config)


def test_exact_deduplication(deduplicator):
    """Test exact deduplication by chunk_id."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
                {"chunk_id": "c2", "text": "text 2", "score": 0.8},
            ]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.85},  # Duplicate
                {"chunk_id": "c3", "text": "text 3", "score": 0.7},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should have 3 unique chunks
    assert len(deduplicated) == 3
    chunk_ids = {r["chunk_id"] for r in deduplicated}
    assert chunk_ids == {"c1", "c2", "c3"}


def test_frequency_tracking(deduplicator):
    """Test frequency tracking for chunks found by multiple variants."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
            ]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.85},
            ]
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.8},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should have 1 chunk with frequency 3
    assert len(deduplicated) == 1
    assert deduplicated[0]["frequency"] == 3
    assert deduplicated[0]["found_by_variants"] == 3


def test_max_score_retention(deduplicator):
    """Test that highest score is retained for duplicates."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.7},
            ]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.95},  # Highest
            ]
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.8},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should retain max score
    assert len(deduplicated) == 1
    assert deduplicated[0]["max_score"] == 0.95


def test_skip_failed_queries(deduplicator):
    """Test that failed queries are skipped."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
            ]
        },
        {
            "variant_index": 1,
            "success": False,  # Failed query
            "results": [],
            "error": "Query failed"
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [
                {"chunk_id": "c2", "text": "text 2", "score": 0.8},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should only have results from successful queries
    assert len(deduplicated) == 2
    chunk_ids = {r["chunk_id"] for r in deduplicated}
    assert chunk_ids == {"c1", "c2"}


def test_empty_results(deduplicator):
    """Test handling of empty results."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": []
        },
        {
            "variant_index": 1,
            "success": True,
            "results": []
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should return empty list
    assert len(deduplicated) == 0


def test_variant_indices_tracking(deduplicator):
    """Test that variant indices are tracked correctly."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
            ]
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.85},
            ]
        },
        {
            "variant_index": 4,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.8},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should track all variant indices
    assert len(deduplicated) == 1
    assert set(deduplicated[0]["variant_indices"]) == {0, 2, 4}
    assert deduplicated[0]["found_by_variants"] == 3


def test_near_duplicate_detection():
    """Test near-duplicate detection using embeddings."""
    # Mock embedding service
    embedding_service = Mock()
    embedding_result = Mock()
    # Create similar embeddings for near-duplicates
    embedding_result.embeddings = [
        [0.1, 0.2, 0.3],  # c1
        [0.11, 0.21, 0.31],  # c2 - very similar to c1
        [0.9, 0.8, 0.7],  # c3 - different
    ]
    embedding_service.embed = Mock(return_value=embedding_result)

    config = MultiQueryConfig(
        enable_near_duplicate_detection=True,
        near_duplicate_threshold=0.95
    )
    deduplicator = ResultDeduplicator(config, embedding_service)

    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
                {"chunk_id": "c2", "text": "text 1 similar", "score": 0.85},
                {"chunk_id": "c3", "text": "different text", "score": 0.8},
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should remove near-duplicate (c2)
    assert len(deduplicated) == 2
    chunk_ids = {r["chunk_id"] for r in deduplicated}
    assert "c1" in chunk_ids
    assert "c3" in chunk_ids
