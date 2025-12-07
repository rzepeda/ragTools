"""
Coherence analyzer for late chunking strategy.

This module analyzes semantic coherence of embedding chunks.
"""

from typing import List, Dict, Any
import logging
import numpy as np

from .models import EmbeddingChunk, CoherenceMetrics

logger = logging.getLogger(__name__)


class CoherenceAnalyzer:
    """Analyze semantic coherence of embedding chunks."""

    def __init__(self, config):
        self.config = config
        self.window_size = config.coherence_window_size

    def analyze_chunk_coherence(
        self,
        chunks: List[EmbeddingChunk]
    ) -> List[EmbeddingChunk]:
        """
        Calculate coherence scores for chunks.

        Args:
            chunks: List of embedding chunks

        Returns:
            Chunks with coherence scores added
        """
        logger.info(f"Analyzing coherence for {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Intra-chunk similarity (would need token embeddings)
            # For now, use a simplified approach
            coherence_score = self._calculate_intra_chunk_coherence(chunk)

            chunk.coherence_score = coherence_score

        return chunks

    def _calculate_intra_chunk_coherence(self, chunk: EmbeddingChunk) -> float:
        """
        Calculate coherence within a chunk.

        In a full implementation, this would analyze token-level embeddings.
        For now, we return a placeholder based on chunk size.
        """
        # Placeholder: In real implementation, would analyze token embeddings
        # within the chunk to measure semantic consistency
        # Higher score for chunks closer to target size
        size_ratio = chunk.token_count / self.config.target_chunk_size
        if size_ratio > 1.0:
            size_ratio = 1.0 / size_ratio
        
        # Base coherence with size penalty
        base_coherence = 0.85
        coherence = base_coherence * (0.7 + 0.3 * size_ratio)
        
        return float(coherence)

    def compare_with_traditional(
        self,
        late_chunks: List[EmbeddingChunk],
        traditional_chunks: List[Any]
    ) -> Dict[str, float]:
        """
        Compare late chunking with traditional chunking.

        Args:
            late_chunks: Chunks from late chunking
            traditional_chunks: Chunks from traditional method

        Returns:
            Comparison metrics
        """
        coherence_scores = [
            c.coherence_score for c in late_chunks
            if c.coherence_score is not None
        ]
        
        metrics = {
            "late_chunking_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "num_late_chunks": len(late_chunks),
            "num_traditional_chunks": len(traditional_chunks),
            "avg_late_chunk_size": float(np.mean([c.token_count for c in late_chunks])),
        }

        logger.info(f"Comparison metrics: {metrics}")

        return metrics
