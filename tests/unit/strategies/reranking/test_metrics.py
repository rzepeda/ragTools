"""Unit tests for ranking metrics."""

import pytest
from rag_factory.strategies.reranking.metrics import (
    dcg_at_k,
    ndcg_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    compare_rankings,
    ranking_correlation,
    RankingAnalyzer,
)


class TestDCG:
    """Tests for DCG (Discounted Cumulative Gain)."""

    def test_dcg_basic(self):
        """Test basic DCG calculation."""
        scores = [3.0, 2.0, 1.0, 0.0]
        dcg = dcg_at_k(scores, 4)

        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
        assert dcg > 0.0

    def test_dcg_empty(self):
        """Test DCG with empty scores."""
        assert dcg_at_k([], 5) == 0.0

    def test_dcg_at_k_limits(self):
        """Test DCG respects k limit."""
        scores = [3.0, 2.0, 1.0, 0.0]

        dcg_k2 = dcg_at_k(scores, 2)
        dcg_k4 = dcg_at_k(scores, 4)

        # DCG@2 should be less than DCG@4
        assert dcg_k2 < dcg_k4


class TestNDCG:
    """Tests for NDCG (Normalized Discounted Cumulative Gain)."""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        scores = [3.0, 2.0, 1.0, 0.0]
        ndcg = ndcg_at_k(scores, 4)

        # Perfect ranking should have NDCG = 1.0
        assert ndcg == pytest.approx(1.0, 0.01)

    def test_ndcg_worst_ranking(self):
        """Test NDCG with worst ranking."""
        scores = [0.0, 1.0, 2.0, 3.0]
        ndcg = ndcg_at_k(scores, 4)

        # Worst ranking should have NDCG < 1.0
        assert ndcg < 1.0

    def test_ndcg_empty(self):
        """Test NDCG with empty scores."""
        assert ndcg_at_k([], 5) == 0.0

    def test_ndcg_all_zeros(self):
        """Test NDCG with all zero scores."""
        scores = [0.0, 0.0, 0.0]
        ndcg = ndcg_at_k(scores, 3)

        assert ndcg == 0.0

    def test_ndcg_range(self):
        """Test NDCG is in valid range."""
        scores = [1.5, 0.8, 2.0, 0.3]
        ndcg = ndcg_at_k(scores, 4)

        assert 0.0 <= ndcg <= 1.0


class TestMRR:
    """Tests for MRR (Mean Reciprocal Rank)."""

    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        mrr_score = mrr([1])
        assert mrr_score == 1.0

    def test_mrr_second_position(self):
        """Test MRR when first relevant result is second."""
        mrr_score = mrr([2])
        assert mrr_score == 0.5

    def test_mrr_third_position(self):
        """Test MRR when first relevant result is third."""
        mrr_score = mrr([3])
        assert mrr_score == pytest.approx(0.333, 0.01)

    def test_mrr_multiple_positions(self):
        """Test MRR takes minimum (best) position."""
        mrr_score = mrr([3, 1, 5])
        assert mrr_score == 1.0  # Takes position 1

    def test_mrr_empty(self):
        """Test MRR with no relevant results."""
        assert mrr([]) == 0.0


class TestPrecisionAtK:
    """Tests for Precision@k."""

    def test_precision_all_relevant(self):
        """Test precision when all results are relevant."""
        relevant = [True, True, True]
        precision = precision_at_k(relevant, 3)

        assert precision == 1.0

    def test_precision_none_relevant(self):
        """Test precision when no results are relevant."""
        relevant = [False, False, False]
        precision = precision_at_k(relevant, 3)

        assert precision == 0.0

    def test_precision_half_relevant(self):
        """Test precision with mixed results."""
        relevant = [True, False, True, False]
        precision = precision_at_k(relevant, 4)

        assert precision == 0.5

    def test_precision_at_k_limits(self):
        """Test precision respects k limit."""
        relevant = [True, True, False, False]
        precision_k2 = precision_at_k(relevant, 2)

        assert precision_k2 == 1.0  # Only first 2 (both True)

    def test_precision_empty(self):
        """Test precision with empty list."""
        assert precision_at_k([], 5) == 0.0


class TestRecallAtK:
    """Tests for Recall@k."""

    def test_recall_all_found(self):
        """Test recall when all relevant items are found."""
        relevant = [True, True, True, False]
        recall = recall_at_k(relevant, total_relevant=3, k=4)

        assert recall == 1.0

    def test_recall_partial_found(self):
        """Test recall when some relevant items are found."""
        relevant = [True, False, True, False]
        recall = recall_at_k(relevant, total_relevant=4, k=4)

        assert recall == 0.5

    def test_recall_at_k_limits(self):
        """Test recall respects k limit."""
        relevant = [True, True, False, True]
        recall = recall_at_k(relevant, total_relevant=3, k=2)

        assert recall == pytest.approx(0.667, 0.01)  # 2 out of 3

    def test_recall_empty(self):
        """Test recall with empty list."""
        assert recall_at_k([], 5, 5) == 0.0


class TestCompareRankings:
    """Tests for compare_rankings function."""

    def test_compare_improved_ranking(self):
        """Test comparison shows improvement."""
        original = [0.5, 0.9, 0.7]  # Poorly ordered
        reranked = [0.9, 0.7, 0.5]  # Well ordered

        comparison = compare_rankings(original, reranked, k=3)

        assert comparison["reranked_ndcg"] > comparison["original_ndcg"]
        assert comparison["ndcg_improvement"] > 0

    def test_compare_same_ranking(self):
        """Test comparison with same ranking."""
        scores = [0.9, 0.7, 0.5]

        comparison = compare_rankings(scores, scores, k=3)

        assert comparison["original_ndcg"] == comparison["reranked_ndcg"]
        assert comparison["ndcg_improvement"] == 0.0

    def test_compare_empty(self):
        """Test comparison with empty scores."""
        comparison = compare_rankings([], [], k=5)

        assert comparison["original_ndcg"] == 0.0
        assert comparison["reranked_ndcg"] == 0.0


class TestRankingCorrelation:
    """Tests for ranking correlation."""

    def test_correlation_identical(self):
        """Test correlation of identical rankings."""
        ranks1 = [0, 1, 2, 3, 4]
        ranks2 = [0, 1, 2, 3, 4]

        corr = ranking_correlation(ranks1, ranks2)
        assert corr == 1.0

    def test_correlation_reversed(self):
        """Test correlation of reversed rankings."""
        ranks1 = [0, 1, 2, 3, 4]
        ranks2 = [4, 3, 2, 1, 0]

        corr = ranking_correlation(ranks1, ranks2)
        assert corr == -1.0

    def test_correlation_different_lengths(self):
        """Test correlation with different length lists."""
        ranks1 = [0, 1, 2]
        ranks2 = [0, 1]

        corr = ranking_correlation(ranks1, ranks2)
        assert corr == 0.0

    def test_correlation_empty(self):
        """Test correlation with empty lists."""
        assert ranking_correlation([], []) == 0.0

    def test_correlation_single_item(self):
        """Test correlation with single item."""
        corr = ranking_correlation([0], [0])
        assert corr == 1.0


class TestRankingAnalyzer:
    """Tests for RankingAnalyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes empty."""
        analyzer = RankingAnalyzer()

        stats = analyzer.get_statistics()
        assert stats["num_rankings"] == 0
        assert stats["avg_original_ndcg"] == 0.0

    def test_analyzer_add_ranking(self):
        """Test adding rankings to analyzer."""
        analyzer = RankingAnalyzer()

        original = [0.5, 0.9, 0.7]
        reranked = [0.9, 0.7, 0.5]

        analyzer.add_ranking("query1", original, reranked, k=3)

        stats = analyzer.get_statistics()
        assert stats["num_rankings"] == 1

    def test_analyzer_multiple_rankings(self):
        """Test analyzer with multiple rankings."""
        analyzer = RankingAnalyzer()

        # Add improved ranking
        analyzer.add_ranking("query1", [0.5, 0.9, 0.7], [0.9, 0.7, 0.5], k=3)

        # Add degraded ranking
        analyzer.add_ranking("query2", [0.9, 0.7, 0.5], [0.5, 0.9, 0.7], k=3)

        stats = analyzer.get_statistics()
        assert stats["num_rankings"] == 2
        assert stats["improved_count"] == 1
        assert stats["degraded_count"] == 1

    def test_analyzer_improvement_rate(self):
        """Test improvement rate calculation."""
        analyzer = RankingAnalyzer()

        # Add 3 improved rankings
        for i in range(3):
            analyzer.add_ranking(
                f"query{i}",
                [0.5, 0.9, 0.7],
                [0.9, 0.7, 0.5],
                k=3
            )

        # Add 1 degraded ranking
        analyzer.add_ranking("query3", [0.9, 0.7, 0.5], [0.5, 0.9, 0.7], k=3)

        stats = analyzer.get_statistics()
        assert stats["improvement_rate"] == 0.75  # 3 out of 4

    def test_analyzer_clear(self):
        """Test clearing analyzer."""
        analyzer = RankingAnalyzer()

        analyzer.add_ranking("query1", [0.9, 0.7], [0.9, 0.7], k=2)
        assert analyzer.get_statistics()["num_rankings"] == 1

        analyzer.clear()
        assert analyzer.get_statistics()["num_rankings"] == 0

    def test_analyzer_average_metrics(self):
        """Test average metric calculations."""
        analyzer = RankingAnalyzer()

        # Add identical rankings (NDCG = 1.0)
        analyzer.add_ranking("query1", [1.0, 0.5], [1.0, 0.5], k=2)
        analyzer.add_ranking("query2", [1.0, 0.5], [1.0, 0.5], k=2)

        stats = analyzer.get_statistics()

        assert stats["avg_original_ndcg"] == pytest.approx(1.0, 0.01)
        assert stats["avg_reranked_ndcg"] == pytest.approx(1.0, 0.01)
        assert stats["avg_ndcg_improvement"] == 0.0
