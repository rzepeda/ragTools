"""Result Deduplicator for Multi-Query RAG Strategy."""

from typing import List, Dict, Any, Set
import logging
import numpy as np

from .config import MultiQueryConfig

logger = logging.getLogger(__name__)


class ResultDeduplicator:
    """Deduplicates results from multiple query variants.
    
    This class handles exact deduplication by chunk_id and optional
    near-duplicate detection using embedding similarity.
    """

    def __init__(self, config: MultiQueryConfig, embedding_service=None):
        """Initialize deduplicator.

        Args:
            config: Multi-query configuration
            embedding_service: Optional embedding service for near-duplicate detection
        """
        self.config = config
        self.embedding_service = embedding_service

    def deduplicate(
        self,
        query_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate results across query variants.

        Args:
            query_results: List of result dicts from parallel execution

        Returns:
            Deduplicated list of results with frequency tracking
        """
        logger.info("Deduplicating results from multiple variants")

        # Track unique chunks
        chunk_map: Dict[str, Dict[str, Any]] = {}
        chunk_frequency: Dict[str, int] = {}
        chunk_max_score: Dict[str, float] = {}
        chunk_variant_indices: Dict[str, List[int]] = {}

        # Collect all results
        total_results = 0
        for query_result in query_results:
            if not query_result.get("success", False):
                continue

            variant_index = query_result["variant_index"]
            results = query_result.get("results", [])
            total_results += len(results)

            for result in results:
                chunk_id = result.get("chunk_id") or result.get("id")

                if not chunk_id:
                    logger.warning("Result missing chunk_id, skipping")
                    continue

                score = result.get("score", 0.0)

                # Track first occurrence
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = result
                    chunk_frequency[chunk_id] = 1
                    chunk_max_score[chunk_id] = score
                    chunk_variant_indices[chunk_id] = [variant_index]
                else:
                    # Update frequency and max score
                    chunk_frequency[chunk_id] += 1
                    chunk_max_score[chunk_id] = max(chunk_max_score[chunk_id], score)
                    if variant_index not in chunk_variant_indices[chunk_id]:
                        chunk_variant_indices[chunk_id].append(variant_index)

        # Near-duplicate detection (optional)
        if self.config.enable_near_duplicate_detection and self.embedding_service:
            chunk_map = self._remove_near_duplicates(chunk_map, chunk_max_score)

        # Build deduplicated results with metadata
        deduplicated = []
        for chunk_id, chunk in chunk_map.items():
            deduplicated.append({
                **chunk,
                "frequency": chunk_frequency[chunk_id],
                "max_score": chunk_max_score[chunk_id],
                "variant_indices": chunk_variant_indices[chunk_id],
                "found_by_variants": len(set(chunk_variant_indices[chunk_id]))
            })

        logger.info(
            f"Deduplicated {total_results} results to {len(deduplicated)} unique chunks"
        )

        return deduplicated

    def _remove_near_duplicates(
        self,
        chunk_map: Dict[str, Dict[str, Any]],
        chunk_max_score: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Remove near-duplicate chunks based on embedding similarity.
        
        Args:
            chunk_map: Map of chunk_id to chunk data
            chunk_max_score: Map of chunk_id to max score
            
        Returns:
            Updated chunk_map with near-duplicates removed
        """
        logger.info("Detecting near-duplicates using embeddings")

        chunk_ids = list(chunk_map.keys())

        if len(chunk_ids) < 2:
            return chunk_map

        # Get embeddings for all chunks
        texts = [chunk_map[cid].get("text", "") for cid in chunk_ids]

        try:
            # Embed all texts
            if hasattr(self.embedding_service, 'embed'):
                embedding_result = self.embedding_service.embed(texts)
                embeddings = embedding_result.embeddings
            elif hasattr(self.embedding_service, 'embed_documents'):
                embeddings = self.embedding_service.embed_documents(texts)
            else:
                logger.warning("Embedding service has no embed method, skipping near-duplicate detection")
                return chunk_map

            # Find near-duplicates
            to_remove: Set[str] = set()

            for i in range(len(chunk_ids)):
                if chunk_ids[i] in to_remove:
                    continue

                for j in range(i + 1, len(chunk_ids)):
                    if chunk_ids[j] in to_remove:
                        continue

                    # Calculate similarity
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                    if similarity >= self.config.near_duplicate_threshold:
                        # Keep the one with higher score
                        if chunk_max_score[chunk_ids[i]] >= chunk_max_score[chunk_ids[j]]:
                            to_remove.add(chunk_ids[j])
                            logger.debug(
                                f"Removing near-duplicate: {chunk_ids[j]} "
                                f"(similar to {chunk_ids[i]}, similarity={similarity:.3f})"
                            )
                        else:
                            to_remove.add(chunk_ids[i])
                            logger.debug(
                                f"Removing near-duplicate: {chunk_ids[i]} "
                                f"(similar to {chunk_ids[j]}, similarity={similarity:.3f})"
                            )
                            break  # Move to next i

            # Remove near-duplicates
            for chunk_id in to_remove:
                del chunk_map[chunk_id]

            logger.info(f"Removed {len(to_remove)} near-duplicate chunks")

        except Exception as e:
            logger.warning(f"Near-duplicate detection failed: {e}, skipping")

        return chunk_map

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
            
        return float(dot_product / norm_product)
