"""
Cross-encoder implementation for reranking.

This module provides a reranker using cross-encoder models from sentence-transformers,
which jointly encode query and document for accurate relevance scoring.
"""

from typing import List, Tuple, Optional
import logging

try:
    from sentence_transformers import CrossEncoder
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import IReranker, RerankConfig

logger = logging.getLogger(__name__)


class CrossEncoderReranker(IReranker):
    """
    Re-ranker using cross-encoder models from sentence-transformers.

    Cross-encoders jointly encode query and document for better relevance scoring
    compared to bi-encoders used in vector search.

    Example:
        >>> config = RerankConfig(model_name="ms-marco-MiniLM-L-6-v2")
        >>> reranker = CrossEncoderReranker(config)
        >>> results = reranker.rerank(query, documents)
    """

    # Popular cross-encoder models
    MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    }

    def __init__(self, config: RerankConfig):
        """
        Initialize cross-encoder reranker.

        Args:
            config: Reranking configuration

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__(config)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and torch are required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers torch\n\n"
                "For lightweight deployment without PyTorch (~2.5GB), consider using:\n"
                "  - CohereReranker (API-based, high quality)\n"
                "  - CosineReranker (local, no dependencies)"
            )

        # Get model name
        model_name = config.model_name
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading cross-encoder model: {model_name} on {device}")

        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

        logger.info(f"Cross-encoder model loaded successfully: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores (not used by cross-encoder)

        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance

        Raises:
            ValueError: If inputs are invalid
        """
        self.validate_inputs(query, documents)

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Predict scores in batches
        batch_size = self.config.batch_size
        all_scores = []

        logger.debug(f"Re-ranking {len(documents)} documents in batches of {batch_size}")

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores.tolist())

        # Create (index, score) pairs and sort by score descending
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranking complete. Top score: {indexed_scores[0][1]:.4f}")

        return indexed_scores

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
