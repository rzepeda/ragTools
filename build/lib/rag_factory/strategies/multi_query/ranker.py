"""Result Ranker for Multi-Query RAG Strategy."""

from typing import List, Dict, Any
import logging

from .config import MultiQueryConfig, RankingStrategy

logger = logging.getLogger(__name__)


class ResultRanker:
    """Ranks merged results from multiple query variants.
    
    This class implements multiple ranking strategies to combine and rank
    results from different query variants.
    """

    def __init__(self, config: MultiQueryConfig):
        """Initialize result ranker.

        Args:
            config: Multi-query configuration
        """
        self.config = config

    def rank(self, deduplicated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank deduplicated results using configured strategy.

        Args:
            deduplicated_results: Deduplicated results with frequency metadata

        Returns:
            Ranked and scored results
        """
        logger.info(
            f"Ranking {len(deduplicated_results)} results using "
            f"{self.config.ranking_strategy.value}"
        )

        if not deduplicated_results:
            return []

        # Apply ranking strategy
        if self.config.ranking_strategy == RankingStrategy.MAX_SCORE:
            ranked = self._rank_by_max_score(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.AVERAGE_SCORE:
            ranked = self._rank_by_average_score(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.FREQUENCY_BOOST:
            ranked = self._rank_by_frequency_boost(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.RECIPROCAL_RANK_FUSION:
            ranked = self._rank_by_rrf(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.HYBRID:
            ranked = self._rank_hybrid(deduplicated_results)
        else:
            # Default to max score
            logger.warning(
                f"Unknown ranking strategy: {self.config.ranking_strategy}, "
                f"using MAX_SCORE"
            )
            ranked = self._rank_by_max_score(deduplicated_results)

        # Sort by final score
        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        # Select top-k
        top_k = ranked[:self.config.final_top_k]

        logger.info(f"Returning top {len(top_k)} results")

        return top_k

    def _rank_by_max_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by maximum score across variants.
        
        Args:
            results: List of deduplicated results
            
        Returns:
            Results with final_score set to max_score
        """
        for result in results:
            result["final_score"] = result.get("max_score", 0.0)
            result["ranking_method"] = "max_score"
        return results

    def _rank_by_average_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by average score (simplified - uses max score as proxy).
        
        Note: This is simplified. In a full implementation, we would track
        all scores per variant and compute true average.
        
        Args:
            results: List of deduplicated results
            
        Returns:
            Results with final_score set
        """
        for result in results:
            # Simplified: use max_score as proxy for average
            # In practice, would need to track all scores
            result["final_score"] = result.get("max_score", 0.0)
            result["ranking_method"] = "average_score"
        return results

    def _rank_by_frequency_boost(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank with frequency boost.
        
        Chunks found by multiple variants are ranked higher.
        
        Args:
            results: List of deduplicated results
            
        Returns:
            Results with final_score boosted by frequency
        """
        for result in results:
            base_score = result.get("max_score", 0.0)
            frequency = result.get("frequency", 1)

            # Boost score based on frequency
            frequency_boost = 1.0 + (frequency - 1) * self.config.frequency_boost_weight
            result["final_score"] = base_score * frequency_boost
            result["ranking_method"] = "frequency_boost"
            result["frequency_boost_factor"] = frequency_boost

        return results

    def _rank_by_rrf(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for all variants
        where rank_i is the rank of this document in variant i
        
        Args:
            results: List of deduplicated results
            
        Returns:
            Results with final_score computed using RRF
        """
        # Group results by variant
        variant_rankings: Dict[int, List[Dict[str, Any]]] = {}

        for result in results:
            for variant_idx in result.get("variant_indices", []):
                if variant_idx not in variant_rankings:
                    variant_rankings[variant_idx] = []
                variant_rankings[variant_idx].append(result)

        # Sort each variant's results by score
        for variant_idx in variant_rankings:
            variant_rankings[variant_idx].sort(
                key=lambda x: x.get("max_score", 0.0),
                reverse=True
            )

        # Calculate RRF scores
        chunk_rrf_scores: Dict[str, float] = {}

        for variant_idx, variant_results in variant_rankings.items():
            for rank, result in enumerate(variant_results, start=1):
                chunk_id = result.get("chunk_id") or result.get("id")
                if not chunk_id:
                    continue
                    
                rrf_contribution = 1.0 / (self.config.rrf_k + rank)

                if chunk_id not in chunk_rrf_scores:
                    chunk_rrf_scores[chunk_id] = 0.0
                chunk_rrf_scores[chunk_id] += rrf_contribution

        # Assign RRF scores to results
        for result in results:
            chunk_id = result.get("chunk_id") or result.get("id")
            result["final_score"] = chunk_rrf_scores.get(chunk_id, 0.0)
            result["ranking_method"] = "reciprocal_rank_fusion"

        return results

    def _rank_hybrid(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hybrid ranking combining RRF and frequency boost.
        
        Args:
            results: List of deduplicated results
            
        Returns:
            Results with final_score computed using hybrid method
        """
        # Apply RRF
        results = self._rank_by_rrf(results)

        # Apply frequency boost on top of RRF
        for result in results:
            rrf_score = result["final_score"]
            frequency = result.get("frequency", 1)
            frequency_boost = 1.0 + (frequency - 1) * self.config.frequency_boost_weight

            result["final_score"] = rrf_score * frequency_boost
            result["ranking_method"] = "hybrid_rrf_frequency"
            result["frequency_boost_factor"] = frequency_boost

        return results
