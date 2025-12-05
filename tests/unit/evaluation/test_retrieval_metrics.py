"""
Unit tests for retrieval metrics.

Tests all retrieval metrics including Precision@K, Recall@K, MRR, NDCG, and Hit Rate.
"""

import pytest
from rag_factory.evaluation.metrics.retrieval import (
    PrecisionAtK,
    RecallAtK,
    MeanReciprocalRank,
    NDCG,
    HitRateAtK,
)


class TestPrecisionAtK:
    """Tests for Precision@K metric."""

    def test_perfect_precision(self):
        """Test perfect precision (all retrieved are relevant)."""
        metric = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 1.0
        assert result.metadata["relevant_in_top_k"] == 5

    def test_zero_precision(self):
        """Test zero precision (no retrieved are relevant)."""
        metric = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc6", "doc7", "doc8"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.0
        assert result.metadata["relevant_in_top_k"] == 0

    def test_partial_precision(self):
        """Test partial precision."""
        metric = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.6  # 3/5
        assert result.metadata["relevant_in_top_k"] == 3

    def test_fewer_retrieved_than_k(self):
        """Test with fewer results than k."""
        metric = PrecisionAtK(k=10)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}

        result = metric.compute(retrieved, relevant)

        # Should only consider first k (which is only 3 here)
        assert result.value == pytest.approx(0.2)  # 2/10

    def test_invalid_k(self):
        """Test that invalid k raises error."""
        with pytest.raises(ValueError):
            PrecisionAtK(k=0)

        with pytest.raises(ValueError):
            PrecisionAtK(k=-1)


class TestRecallAtK:
    """Tests for Recall@K metric."""

    def test_perfect_recall(self):
        """Test perfect recall (all relevant are retrieved)."""
        metric = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 1.0
        assert result.metadata["relevant_in_top_k"] == 3

    def test_zero_recall(self):
        """Test zero recall (no relevant retrieved)."""
        metric = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5", "doc6"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.0

    def test_partial_recall(self):
        """Test partial recall."""
        metric = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6", "doc7"}  # 4 total relevant

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.5  # 2/4 relevant docs retrieved

    def test_empty_relevant(self):
        """Test with no relevant documents."""
        metric = RecallAtK(k=5)
        retrieved = ["doc1", "doc2"]
        relevant = set()

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.0
        assert "warning" in result.metadata


class TestMeanReciprocalRank:
    """Tests for Mean Reciprocal Rank metric."""

    def test_first_relevant(self):
        """Test when first result is relevant."""
        metric = MeanReciprocalRank()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 1.0
        assert result.metadata["first_relevant_rank"] == 1

    def test_third_relevant(self):
        """Test when third result is relevant."""
        metric = MeanReciprocalRank()
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc3"}

        result = metric.compute(retrieved, relevant)

        assert result.value == pytest.approx(1/3)
        assert result.metadata["first_relevant_rank"] == 3

    def test_no_relevant_found(self):
        """Test when no relevant documents found."""
        metric = MeanReciprocalRank()
        retrieved = ["doc1", "doc2"]
        relevant = {"doc3"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.0
        assert result.metadata["first_relevant_rank"] is None

    def test_multiple_relevant(self):
        """Test with multiple relevant (should use first)."""
        metric = MeanReciprocalRank()
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc2", "doc4"}

        result = metric.compute(retrieved, relevant)

        assert result.value == pytest.approx(1/2)  # First relevant is at position 2
        assert result.metadata["first_relevant_rank"] == 2


class TestNDCG:
    """Tests for NDCG metric."""

    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        metric = NDCG(k=3)
        retrieved = ["doc1", "doc2", "doc3"]
        relevance_scores = {
            "doc1": 3,  # Most relevant first
            "doc2": 2,
            "doc3": 1
        }

        result = metric.compute(retrieved, relevance_scores)

        # Perfect ranking should give NDCG = 1.0
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_worst_ranking(self):
        """Test NDCG with reversed ranking."""
        metric = NDCG(k=3)
        retrieved = ["doc3", "doc2", "doc1"]
        relevance_scores = {
            "doc1": 3,  # Most relevant last
            "doc2": 2,
            "doc3": 1
        }

        result = metric.compute(retrieved, relevance_scores)

        # Worst ranking should give NDCG < 1.0
        assert 0 <= result.value < 1.0

    def test_no_relevant_docs(self):
        """Test NDCG when no documents are relevant."""
        metric = NDCG(k=3)
        retrieved = ["doc1", "doc2", "doc3"]
        relevance_scores = {}

        result = metric.compute(retrieved, relevance_scores)

        assert result.value == 0.0

    def test_partial_relevance(self):
        """Test NDCG with some relevant, some not."""
        metric = NDCG(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevance_scores = {
            "doc1": 3,
            "doc2": 0,  # Not relevant
            "doc3": 2,
            "doc4": 1,
            "doc5": 0   # Not relevant
        }

        result = metric.compute(retrieved, relevance_scores)

        assert 0 <= result.value <= 1.0
        assert "dcg" in result.metadata
        assert "idcg" in result.metadata


class TestHitRateAtK:
    """Tests for Hit Rate@K metric."""

    def test_hit(self):
        """Test when at least one relevant doc is found."""
        metric = HitRateAtK(k=10)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 1.0
        assert result.metadata["has_hit"] is True

    def test_no_hit(self):
        """Test when no relevant docs are found."""
        metric = HitRateAtK(k=10)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}

        result = metric.compute(retrieved, relevant)

        assert result.value == 0.0
        assert result.metadata["has_hit"] is False

    def test_hit_outside_k(self):
        """Test when relevant doc is beyond k."""
        metric = HitRateAtK(k=2)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}  # Beyond k=2

        result = metric.compute(retrieved, relevant)

        # Should not count doc3 since it's at position 3
        assert result.value == 0.0


class TestMetricProperties:
    """Test common properties of metrics."""

    def test_metric_names(self):
        """Test that metric names are correctly set."""
        assert PrecisionAtK(k=5).name == "precision@5"
        assert RecallAtK(k=10).name == "recall@10"
        assert MeanReciprocalRank().name == "mrr"
        assert NDCG(k=5).name == "ndcg@5"
        assert HitRateAtK(k=10).name == "hit_rate@10"

    def test_higher_is_better(self):
        """Test that higher_is_better property is correct."""
        assert PrecisionAtK(k=5).higher_is_better is True
        assert RecallAtK(k=5).higher_is_better is True
        assert MeanReciprocalRank().higher_is_better is True
        assert NDCG(k=5).higher_is_better is True
        assert HitRateAtK(k=10).higher_is_better is True
