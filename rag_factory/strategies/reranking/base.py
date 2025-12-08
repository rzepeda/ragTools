"""
Base interface for reranking models.

This module defines the abstract base class and data structures for reranking
retrieved documents to improve relevance scoring.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RerankerModel(Enum):
    """Enumeration of supported re-ranker models."""
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    BGE = "bge"
    COSINE = "cosine"  # Lightweight cosine similarity reranker
    CUSTOM = "custom"


@dataclass
class RerankResult:
    """Result from re-ranking operation."""
    document_id: str
    original_rank: int
    reranked_rank: int
    original_score: float  # Vector similarity score
    rerank_score: float    # Re-ranker relevance score
    normalized_score: float  # Normalized score (0.0-1.0)


@dataclass
class RerankResponse:
    """Response from re-ranking service."""
    query: str
    results: List[RerankResult]
    total_candidates: int
    reranked_count: int
    top_k: int
    model_used: str
    execution_time_ms: float
    cache_hit: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankConfig:
    """Configuration for re-ranking."""
    model: RerankerModel = RerankerModel.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval settings
    initial_retrieval_size: int = 100  # Number of candidates to retrieve
    top_k: int = 10  # Number of results to return after re-ranking

    # Scoring settings
    score_threshold: float = 0.0  # Minimum score to include
    normalize_scores: bool = True

    # Performance settings
    batch_size: int = 32
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    timeout_seconds: float = 5.0

    # Fallback settings
    enable_fallback: bool = True
    fallback_to_vector_scores: bool = True

    # Model-specific config
    model_config: Dict[str, Any] = field(default_factory=dict)


class IReranker(ABC):
    """Abstract base class for re-ranking models."""

    def __init__(self, config: RerankConfig):
        """Initialize re-ranker with configuration."""
        self.config = config

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents based on query relevance.

        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores from vector search

        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the re-ranker model."""
        pass

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0.0-1.0 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def validate_inputs(self, query: str, documents: List[str]) -> None:
        """Validate inputs to re-ranker."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if len(documents) > 500:
            raise ValueError(f"Too many documents: {len(documents)} (max: 500)")
