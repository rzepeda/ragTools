"""Unit tests for ResultRanker."""

import pytest
from rag_factory.strategies.multi_query.ranker import ResultRanker
from rag_factory.strategies.multi_query.config import MultiQueryConfig, RankingStrategy


@pytest.fixture
def config():
    """Create default config."""
    return MultiQueryConfig()


@pytest.fixture
def ranker(config):
    """Create ranker instance."""
    return ResultRanker(config)


def test_rank_by_max_score():
    """Test ranking by maximum score."""
    config = MultiQueryConfig(ranking_strategy=RankingStrategy.MAX_SCORE, final_top_k=3)
    ranker = ResultRanker(config)

    results = [
        {"chunk_id": "c1", "max_score": 0.9, "frequency": 2},
        {"chunk_id": "c2", "max_score": 0.7, "frequency": 1},
        {"chunk_id": "c3", "max_score": 0.95, "frequency": 1},
    ]

    ranked = ranker.rank(results)

    # Should be sorted by max_score
    assert len(ranked) == 3
    assert ranked[0]["chunk_id"] == "c3"  # Highest score
    assert ranked[1]["chunk_id"] == "c1"
    assert ranked[2]["chunk_id"] == "c2"
    assert all(r["ranking_method"] == "max_score" for r in ranked)


def test_rank_by_frequency_boost():
    """Test ranking with frequency boost."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.FREQUENCY_BOOST,
        frequency_boost_weight=0.2,
        final_top_k=3
    )
    ranker = ResultRanker(config)

    results = [
        {"chunk_id": "c1", "max_score": 0.8, "frequency": 3},  # Lower score but high frequency
        {"chunk_id": "c2", "max_score": 0.9, "frequency": 1},  # High score but low frequency
        {"chunk_id": "c3", "max_score": 0.85, "frequency": 2},
    ]

    ranked = ranker.rank(results)

    # c1 should be boosted due to frequency
    # c1: 0.8 * (1 + (3-1)*0.2) = 0.8 * 1.4 = 1.12
    # c2: 0.9 * (1 + (1-1)*0.2) = 0.9 * 1.0 = 0.9
    # c3: 0.85 * (1 + (2-1)*0.2) = 0.85 * 1.2 = 1.02

    assert ranked[0]["chunk_id"] == "c1"  # Highest after boost
    assert ranked[0]["ranking_method"] == "frequency_boost"
    assert "frequency_boost_factor" in ranked[0]


def test_rank_by_rrf():
    """Test Reciprocal Rank Fusion ranking."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
        rrf_k=60,
        final_top_k=3
    )
    ranker = ResultRanker(config)

    results = [
        {
            "chunk_id": "c1",
            "max_score": 0.9,
            "frequency": 2,
            "variant_indices": [0, 1]
        },
        {
            "chunk_id": "c2",
            "max_score": 0.8,
            "frequency": 1,
            "variant_indices": [0]
        },
        {
            "chunk_id": "c3",
            "max_score": 0.85,
            "frequency": 1,
            "variant_indices": [1]
        },
    ]

    ranked = ranker.rank(results)

    # Should have RRF scores
    assert len(ranked) == 3
    assert all(r["ranking_method"] == "reciprocal_rank_fusion" for r in ranked)
    assert all("final_score" in r for r in ranked)


def test_top_k_selection():
    """Test top-k selection."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.MAX_SCORE,
        final_top_k=2  # Only return top 2
    )
    ranker = ResultRanker(config)

    results = [
        {"chunk_id": "c1", "max_score": 0.9},
        {"chunk_id": "c2", "max_score": 0.8},
        {"chunk_id": "c3", "max_score": 0.95},
        {"chunk_id": "c4", "max_score": 0.7},
    ]

    ranked = ranker.rank(results)

    # Should only return top 2
    assert len(ranked) == 2
    assert ranked[0]["chunk_id"] == "c3"
    assert ranked[1]["chunk_id"] == "c1"


def test_empty_results(ranker):
    """Test handling of empty results."""
    ranked = ranker.rank([])

    assert ranked == []


def test_hybrid_ranking():
    """Test hybrid ranking (RRF + frequency boost)."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.HYBRID,
        rrf_k=60,
        frequency_boost_weight=0.2,
        final_top_k=3
    )
    ranker = ResultRanker(config)

    results = [
        {
            "chunk_id": "c1",
            "max_score": 0.9,
            "frequency": 3,
            "variant_indices": [0, 1, 2]
        },
        {
            "chunk_id": "c2",
            "max_score": 0.95,
            "frequency": 1,
            "variant_indices": [0]
        },
        {
            "chunk_id": "c3",
            "max_score": 0.85,
            "frequency": 2,
            "variant_indices": [1, 2]
        },
    ]

    ranked = ranker.rank(results)

    # Should combine RRF with frequency boost
    assert len(ranked) == 3
    assert all(r["ranking_method"] == "hybrid_rrf_frequency" for r in ranked)
    assert all("frequency_boost_factor" in r for r in ranked)


def test_ranking_preserves_metadata():
    """Test that ranking preserves original metadata."""
    config = MultiQueryConfig(ranking_strategy=RankingStrategy.MAX_SCORE, final_top_k=2)
    ranker = ResultRanker(config)

    results = [
        {
            "chunk_id": "c1",
            "text": "text 1",
            "max_score": 0.9,
            "frequency": 2,
            "custom_field": "value1"
        },
        {
            "chunk_id": "c2",
            "text": "text 2",
            "max_score": 0.8,
            "frequency": 1,
            "custom_field": "value2"
        },
    ]

    ranked = ranker.rank(results)

    # Should preserve all original fields
    assert ranked[0]["text"] == "text 1"
    assert ranked[0]["custom_field"] == "value1"
    assert ranked[1]["text"] == "text 2"
    assert ranked[1]["custom_field"] == "value2"
