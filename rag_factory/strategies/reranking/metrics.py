"""
Ranking metrics for evaluating reranking quality.

This module provides implementations of common ranking metrics like NDCG and MRR
to measure the effectiveness of reranking strategies.
"""

import math
from typing import List, Dict, Any


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    DCG measures the usefulness of a ranking based on the position of relevant items.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top results to consider

    Returns:
        DCG score
    """
    if not relevance_scores:
        return 0.0

    dcg = 0.0
    for i, score in enumerate(relevance_scores[:k]):
        # DCG formula: sum(rel_i / log2(i + 2))
        dcg += score / math.log2(i + 2)

    return dcg


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG normalizes DCG by the ideal DCG (scores sorted by relevance).

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top results to consider

    Returns:
        NDCG score between 0.0 and 1.0
    """
    if not relevance_scores:
        return 0.0

    dcg = dcg_at_k(relevance_scores, k)

    # Ideal DCG: sort scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


def mrr(relevant_positions: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR measures how quickly the first relevant result appears.

    Args:
        relevant_positions: List of positions (1-indexed) where relevant items appear

    Returns:
        MRR score between 0.0 and 1.0
    """
    if not relevant_positions:
        return 0.0

    # Take the first relevant position (lowest rank)
    first_relevant = min(relevant_positions)

    return 1.0 / first_relevant


def precision_at_k(relevant_items: List[bool], k: int) -> float:
    """
    Calculate Precision at k.

    Precision@k measures the proportion of relevant items in top-k results.

    Args:
        relevant_items: List of boolean values indicating relevance
        k: Number of top results to consider

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not relevant_items or k <= 0:
        return 0.0

    top_k = relevant_items[:k]
    return sum(top_k) / len(top_k)


def recall_at_k(relevant_items: List[bool], total_relevant: int, k: int) -> float:
    """
    Calculate Recall at k.

    Recall@k measures the proportion of all relevant items found in top-k results.

    Args:
        relevant_items: List of boolean values indicating relevance
        total_relevant: Total number of relevant items in the dataset
        k: Number of top results to consider

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not relevant_items or k <= 0 or total_relevant <= 0:
        return 0.0

    top_k = relevant_items[:k]
    return sum(top_k) / total_relevant


def compare_rankings(
    original_scores: List[float],
    reranked_scores: List[float],
    k: int = 10
) -> Dict[str, Any]:
    """
    Compare original and reranked results using multiple metrics.

    Args:
        original_scores: Relevance scores in original ranking order
        reranked_scores: Relevance scores in reranked order
        k: Number of top results to consider

    Returns:
        Dictionary with comparison metrics
    """
    metrics = {
        "original_ndcg": ndcg_at_k(original_scores, k),
        "reranked_ndcg": ndcg_at_k(reranked_scores, k),
        "ndcg_improvement": 0.0,
        "position_changes": 0,
        "avg_position_change": 0.0
    }

    # Calculate NDCG improvement
    if metrics["original_ndcg"] > 0:
        metrics["ndcg_improvement"] = (
            (metrics["reranked_ndcg"] - metrics["original_ndcg"]) /
            metrics["original_ndcg"]
        )

    return metrics


def ranking_correlation(original_ranks: List[int], reranked_ranks: List[int]) -> float:
    """
    Calculate Spearman's rank correlation coefficient.

    Measures how similar two rankings are.

    Args:
        original_ranks: Original ranking positions (0-indexed)
        reranked_ranks: Reranked positions (0-indexed)

    Returns:
        Correlation coefficient between -1.0 and 1.0
    """
    if not original_ranks or len(original_ranks) != len(reranked_ranks):
        return 0.0

    n = len(original_ranks)
    if n < 2:
        return 1.0

    # Calculate sum of squared differences
    d_squared_sum = sum((o - r) ** 2 for o, r in zip(original_ranks, reranked_ranks))

    # Spearman correlation formula
    correlation = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

    return correlation


class RankingAnalyzer:
    """
    Analyzer for tracking and comparing ranking changes.

    Example:
        >>> analyzer = RankingAnalyzer()
        >>> analyzer.add_ranking(original_scores, reranked_scores)
        >>> stats = analyzer.get_statistics()
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.rankings = []

    def add_ranking(
        self,
        query: str,
        original_scores: List[float],
        reranked_scores: List[float],
        k: int = 10
    ) -> None:
        """
        Add a ranking comparison.

        Args:
            query: The search query
            original_scores: Original ranking scores
            reranked_scores: Reranked scores
            k: Number of top results to analyze
        """
        comparison = compare_rankings(original_scores, reranked_scores, k)
        comparison["query"] = query

        self.rankings.append(comparison)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all rankings.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.rankings:
            return {
                "num_rankings": 0,
                "avg_original_ndcg": 0.0,
                "avg_reranked_ndcg": 0.0,
                "avg_ndcg_improvement": 0.0,
                "improved_count": 0,
                "degraded_count": 0,
                "unchanged_count": 0
            }

        num_rankings = len(self.rankings)
        avg_original_ndcg = sum(r["original_ndcg"] for r in self.rankings) / num_rankings
        avg_reranked_ndcg = sum(r["reranked_ndcg"] for r in self.rankings) / num_rankings
        avg_improvement = sum(r["ndcg_improvement"] for r in self.rankings) / num_rankings

        improved = sum(1 for r in self.rankings if r["ndcg_improvement"] > 0)
        degraded = sum(1 for r in self.rankings if r["ndcg_improvement"] < 0)
        unchanged = num_rankings - improved - degraded

        return {
            "num_rankings": num_rankings,
            "avg_original_ndcg": avg_original_ndcg,
            "avg_reranked_ndcg": avg_reranked_ndcg,
            "avg_ndcg_improvement": avg_improvement,
            "improved_count": improved,
            "degraded_count": degraded,
            "unchanged_count": unchanged,
            "improvement_rate": improved / num_rankings if num_rankings > 0 else 0.0
        }

    def clear(self) -> None:
        """Clear all stored rankings."""
        self.rankings.clear()
