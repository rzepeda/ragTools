"""
Reranker service for orchestrating document reranking.

This module provides the main service interface for reranking documents,
handling caching, fallback, and performance tracking.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging

from .base import IReranker, RerankConfig, RerankResponse, RerankResult, RerankerModel
from .cross_encoder_reranker import CrossEncoderReranker
from .cohere_reranker import CohereReranker
from .bge_reranker import BGEReranker
from .cache import RerankCache

logger = logging.getLogger(__name__)


@dataclass
class CandidateDocument:
    """Document candidate for re-ranking."""
    id: str
    text: str
    original_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RerankerService:
    """
    Service for re-ranking retrieved documents using various models.

    This service handles the two-step retrieval process: broad initial retrieval
    followed by precise reranking to return the most relevant results.

    Example:
        >>> config = RerankConfig(
        ...     model=RerankerModel.CROSS_ENCODER,
        ...     initial_retrieval_size=100,
        ...     top_k=10
        ... )
        >>> service = RerankerService(config)
        >>> response = service.rerank(query, candidates)
    """

    def __init__(self, config: RerankConfig):
        """
        Initialize reranker service.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.reranker = self._init_reranker()
        self.cache = RerankCache(config) if config.enable_cache else None
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_documents_reranked": 0,
            "avg_execution_time_ms": 0.0
        }

        logger.info(f"RerankerService initialized with model: {self.reranker.get_model_name()}")

    def _init_reranker(self) -> IReranker:
        """Initialize the re-ranker based on configuration."""
        reranker_map = {
            RerankerModel.CROSS_ENCODER: CrossEncoderReranker,
            RerankerModel.COHERE: CohereReranker,
            RerankerModel.BGE: BGEReranker
        }

        reranker_class = reranker_map.get(self.config.model)
        if not reranker_class:
            raise ValueError(f"Unknown re-ranker model: {self.config.model}")

        return reranker_class(self.config)

    def rerank(
        self,
        query: str,
        candidates: List[CandidateDocument]
    ) -> RerankResponse:
        """
        Re-rank candidate documents for a query.

        Args:
            query: The search query
            candidates: List of candidate documents with original scores

        Returns:
            RerankResponse with re-ranked results
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        # Validate inputs
        if not candidates:
            return self._empty_response(query)

        # Check cache
        cache_hit = False
        cache_key = None
        if self.cache is not None:
            cache_key = self._compute_cache_key(query, candidates)
            cached_response = self.cache.get(cache_key)

            if cached_response:
                self._stats["cache_hits"] += 1
                # Create a new response object with cache_hit=True
                from dataclasses import replace
                cached_response_with_flag = replace(cached_response, cache_hit=True)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_response_with_flag
            else:
                self._stats["cache_misses"] += 1
        else:
            self._stats["cache_misses"] += 1

        # Extract document texts and scores
        documents = [c.text for c in candidates]
        original_scores = [c.original_score for c in candidates]

        try:
            # Perform re-ranking
            logger.debug(f"Re-ranking {len(candidates)} candidates for query: {query[:50]}...")
            reranked_indices_scores = self.reranker.rerank(
                query,
                documents,
                original_scores
            )

            # Apply score threshold
            if self.config.score_threshold > 0:
                reranked_indices_scores = [
                    (idx, score) for idx, score in reranked_indices_scores
                    if score >= self.config.score_threshold
                ]

            # Limit to top-k
            reranked_indices_scores = reranked_indices_scores[:self.config.top_k]

            # Normalize scores if configured
            rerank_scores = [score for _, score in reranked_indices_scores]
            if self.config.normalize_scores:
                normalized = self.reranker.normalize_scores(rerank_scores)
            else:
                normalized = rerank_scores

            # Build results
            results = []
            for new_rank, ((orig_idx, rerank_score), norm_score) in enumerate(
                zip(reranked_indices_scores, normalized)
            ):
                candidate = candidates[orig_idx]
                result = RerankResult(
                    document_id=candidate.id,
                    original_rank=orig_idx,
                    reranked_rank=new_rank,
                    original_score=candidate.original_score,
                    rerank_score=rerank_score,
                    normalized_score=norm_score
                )
                results.append(result)

            execution_time_ms = (time.time() - start_time) * 1000

            response = RerankResponse(
                query=query,
                results=results,
                total_candidates=len(candidates),
                reranked_count=len(results),
                top_k=self.config.top_k,
                model_used=self.reranker.get_model_name(),
                execution_time_ms=execution_time_ms,
                cache_hit=cache_hit
            )

            # Cache the response
            if self.cache is not None and cache_key:
                try:
                    self.cache.set(cache_key, response)
                    logger.debug(f"Cached response for key: {cache_key[:16]}... (cache size: {len(self.cache)})")
                except Exception as cache_error:
                    logger.error(f"Failed to cache response: {cache_error}", exc_info=True)

            # Update stats
            self._stats["total_documents_reranked"] += len(candidates)
            self._update_avg_execution_time(execution_time_ms)

            logger.info(
                f"Re-ranking complete: {len(results)} results in {execution_time_ms:.2f}ms "
                f"(model: {self.reranker.get_model_name()})"
            )

            return response

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}", exc_info=True)

            # Fallback to original ranking if configured
            if self.config.enable_fallback and self.config.fallback_to_vector_scores:
                logger.warning("Falling back to vector similarity scores")
                return self._fallback_ranking(query, candidates, start_time)
            else:
                raise

    def _fallback_ranking(
        self,
        query: str,
        candidates: List[CandidateDocument],
        start_time: float
    ) -> RerankResponse:
        """Fallback to vector similarity ranking."""
        # Sort by original scores
        sorted_candidates = sorted(
            enumerate(candidates),
            key=lambda x: x[1].original_score,
            reverse=True
        )[:self.config.top_k]

        results = []
        for new_rank, (orig_idx, candidate) in enumerate(sorted_candidates):
            result = RerankResult(
                document_id=candidate.id,
                original_rank=orig_idx,
                reranked_rank=new_rank,
                original_score=candidate.original_score,
                rerank_score=candidate.original_score,
                normalized_score=candidate.original_score
            )
            results.append(result)

        execution_time_ms = (time.time() - start_time) * 1000

        return RerankResponse(
            query=query,
            results=results,
            total_candidates=len(candidates),
            reranked_count=len(results),
            top_k=self.config.top_k,
            model_used="fallback_vector_similarity",
            execution_time_ms=execution_time_ms,
            cache_hit=False,
            metadata={"fallback": True}
        )

    def _empty_response(self, query: str) -> RerankResponse:
        """Return empty response for empty candidate list."""
        return RerankResponse(
            query=query,
            results=[],
            total_candidates=0,
            reranked_count=0,
            top_k=self.config.top_k,
            model_used=self.reranker.get_model_name(),
            execution_time_ms=0.0,
            cache_hit=False
        )

    def _compute_cache_key(self, query: str, candidates: List[CandidateDocument]) -> str:
        """Compute cache key from query and candidates."""
        doc_ids = [c.id for c in candidates]
        return RerankCache.compute_key(query, doc_ids, self.reranker.get_model_name())

    def _update_avg_execution_time(self, new_time_ms: float):
        """Update average execution time."""
        total = self._stats["total_requests"]
        current_avg = self._stats["avg_execution_time_ms"]

        # Incremental average
        new_avg = ((current_avg * (total - 1)) + new_time_ms) / total
        self._stats["avg_execution_time_ms"] = new_avg

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_hit_rate = 0.0
        if self._stats["total_requests"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / self._stats["total_requests"]

        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "model": self.reranker.get_model_name()
        }

    def clear_cache(self):
        """Clear the re-ranking cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Re-ranking cache cleared")
