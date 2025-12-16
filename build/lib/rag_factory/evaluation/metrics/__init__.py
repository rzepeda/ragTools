"""
Evaluation metrics for RAG strategies.

This module provides various metrics for evaluating retrieval and generation
quality, including:
- Retrieval metrics (Precision@K, Recall@K, MRR, NDCG, Hit Rate)
- Quality metrics (Semantic Similarity, Faithfulness, Relevance)
- Performance metrics (Latency, Throughput)
- Cost metrics (Token usage, API costs)
"""

from rag_factory.evaluation.metrics.base import (
    IMetric,
    MetricResult,
    MetricType,
)
from rag_factory.evaluation.metrics.retrieval import (
    PrecisionAtK,
    RecallAtK,
    MeanReciprocalRank,
    NDCG,
    HitRateAtK,
)

__all__ = [
    "IMetric",
    "MetricResult",
    "MetricType",
    "PrecisionAtK",
    "RecallAtK",
    "MeanReciprocalRank",
    "NDCG",
    "HitRateAtK",
]
