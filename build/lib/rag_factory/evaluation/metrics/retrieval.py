"""
Retrieval evaluation metrics.

This module provides metrics for evaluating document retrieval quality,
including Precision@K, Recall@K, Mean Reciprocal Rank (MRR),
Normalized Discounted Cumulative Gain (NDCG), and Hit Rate@K.
"""

from typing import List, Set, Dict, Optional
import numpy as np
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType


class PrecisionAtK(IMetric):
    """
    Precision@K: Proportion of retrieved documents that are relevant.

    Precision@K measures the fraction of retrieved documents in the top K
    that are actually relevant. It focuses on the quality of retrieval.

    Formula: Precision@K = (# relevant docs in top K) / K

    Args:
        k: Number of top results to consider (default: 5)

    Example:
        >>> metric = PrecisionAtK(k=5)
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc5"}
        >>> result = metric.compute(retrieved, relevant)
        >>> print(f"Precision@5: {result.value}")  # 0.6 (3/5)
    """

    def __init__(self, k: int = 5):
        """
        Initialize Precision@K metric.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(f"precision@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute Precision@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            query_id: Optional query identifier

        Returns:
            MetricResult with precision value (0.0 to 1.0)
        """
        top_k = retrieved_ids[:self.k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        precision = relevant_in_top_k / self.k if self.k > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=precision,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_in_top_k,
                "total_relevant": len(relevant_ids),
                "retrieved_count": len(top_k)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Proportion of top-{self.k} retrieved documents that are relevant"


class RecallAtK(IMetric):
    """
    Recall@K: Proportion of relevant documents retrieved in top K.

    Recall@K measures the fraction of all relevant documents that were
    successfully retrieved in the top K results. It focuses on coverage.

    Formula: Recall@K = (# relevant docs in top K) / (# total relevant docs)

    Args:
        k: Number of top results to consider (default: 5)

    Example:
        >>> metric = RecallAtK(k=5)
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc6", "doc7"}  # 4 total relevant
        >>> result = metric.compute(retrieved, relevant)
        >>> print(f"Recall@5: {result.value}")  # 0.5 (2/4)
    """

    def __init__(self, k: int = 5):
        """
        Initialize Recall@K metric.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(f"recall@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute Recall@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            query_id: Optional query identifier

        Returns:
            MetricResult with recall value (0.0 to 1.0)
        """
        if not relevant_ids:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"k": self.k, "total_relevant": 0, "warning": "No relevant documents"},
                query_id=query_id
            )

        top_k = retrieved_ids[:self.k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        recall = relevant_in_top_k / len(relevant_ids)

        return MetricResult(
            name=self.name,
            value=recall,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_in_top_k,
                "total_relevant": len(relevant_ids),
                "retrieved_count": len(top_k)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Proportion of relevant documents retrieved in top-{self.k}"


class MeanReciprocalRank(IMetric):
    """
    Mean Reciprocal Rank (MRR): Reciprocal of rank of first relevant document.

    MRR measures how quickly a relevant document is found. It's the inverse
    of the position of the first relevant result.

    Formula: MRR = 1 / (rank of first relevant document)

    Example:
        >>> metric = MeanReciprocalRank()
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> relevant = {"doc3"}  # First relevant at position 3
        >>> result = metric.compute(retrieved, relevant)
        >>> print(f"MRR: {result.value}")  # 0.333... (1/3)
    """

    def __init__(self):
        """Initialize Mean Reciprocal Rank metric."""
        super().__init__("mrr", MetricType.RETRIEVAL)

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute MRR.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            query_id: Optional query identifier

        Returns:
            MetricResult with MRR value (0.0 to 1.0)
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                return MetricResult(
                    name=self.name,
                    value=mrr,
                    metadata={
                        "first_relevant_rank": rank,
                        "total_retrieved": len(retrieved_ids)
                    },
                    query_id=query_id
                )

        # No relevant document found
        return MetricResult(
            name=self.name,
            value=0.0,
            metadata={
                "first_relevant_rank": None,
                "total_retrieved": len(retrieved_ids),
                "warning": "No relevant documents retrieved"
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Reciprocal rank of first relevant document"


class NDCG(IMetric):
    """
    Normalized Discounted Cumulative Gain (NDCG).

    NDCG considers both relevance grades and position, rewarding relevant
    documents that appear higher in the ranking. It's normalized to be
    between 0 and 1.

    Formula: NDCG@K = DCG@K / IDCG@K
    Where DCG = Σ(2^rel - 1) / log2(rank + 1)

    Args:
        k: Number of top results to consider (default: 10)

    Example:
        >>> metric = NDCG(k=5)
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> relevance_scores = {"doc1": 3, "doc2": 0, "doc3": 2}
        >>> result = metric.compute(retrieved, relevance_scores)
        >>> print(f"NDCG@5: {result.value}")
    """

    def __init__(self, k: int = 10):
        """
        Initialize NDCG metric.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(f"ndcg@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute NDCG@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevance_scores: Dictionary mapping doc IDs to relevance scores (0-3 typical)
            query_id: Optional query identifier

        Returns:
            MetricResult with NDCG value (0.0 to 1.0)
        """
        def dcg(scores: List[float]) -> float:
            """Calculate Discounted Cumulative Gain."""
            return sum(
                (2**score - 1) / np.log2(rank + 2)
                for rank, score in enumerate(scores)
            )

        # Get relevance scores for retrieved docs (pad with 0s if needed)
        retrieved_scores = [
            relevance_scores.get(doc_id, 0.0)
            for doc_id in retrieved_ids[:self.k]
        ]

        # Calculate DCG for retrieved ranking
        dcg_value = dcg(retrieved_scores)

        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:self.k]
        idcg_value = dcg(ideal_scores)

        # Calculate NDCG
        ndcg_value = dcg_value / idcg_value if idcg_value > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=ndcg_value,
            metadata={
                "k": self.k,
                "dcg": dcg_value,
                "idcg": idcg_value,
                "retrieved_count": len(retrieved_scores)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Normalized Discounted Cumulative Gain at top-{self.k}"


class HitRateAtK(IMetric):
    """
    Hit Rate@K: Percentage of queries with at least one relevant document in top K.

    Hit Rate is a binary metric that indicates whether any relevant document
    was found in the top K results. Useful for measuring if the system
    retrieves anything useful.

    Formula: HitRate@K = 1 if any doc in top K is relevant, else 0

    Args:
        k: Number of top results to consider (default: 10)

    Example:
        >>> metric = HitRateAtK(k=10)
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> relevant = {"doc2"}
        >>> result = metric.compute(retrieved, relevant)
        >>> print(f"Hit Rate@10: {result.value}")  # 1.0
    """

    def __init__(self, k: int = 10):
        """
        Initialize Hit Rate@K metric.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(f"hit_rate@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute Hit Rate@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            query_id: Optional query identifier

        Returns:
            MetricResult with hit rate value (0.0 or 1.0)
        """
        top_k = retrieved_ids[:self.k]
        has_hit = any(doc_id in relevant_ids for doc_id in top_k)
        hit_rate = 1.0 if has_hit else 0.0

        return MetricResult(
            name=self.name,
            value=hit_rate,
            metadata={
                "k": self.k,
                "has_hit": has_hit,
                "total_relevant": len(relevant_ids),
                "retrieved_count": len(top_k)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Whether at least one relevant document appears in top-{self.k}"
