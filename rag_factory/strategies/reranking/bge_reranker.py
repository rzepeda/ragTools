"""
BGE (BAAI General Embedding) reranker implementation.

This module provides a reranker using BGE reranker models from BAAI,
which are optimized for Chinese and multilingual reranking tasks.
"""

from typing import List, Tuple, Optional
import logging

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import IReranker, RerankConfig

logger = logging.getLogger(__name__)


class BGEReranker(IReranker):
    """
    Re-ranker using BGE reranker models.

    BGE rerankers are cross-encoder models optimized for reranking tasks,
    with strong performance on both English and Chinese text.

    Example:
        >>> config = RerankConfig(model_name="BAAI/bge-reranker-base")
        >>> reranker = BGEReranker(config)
        >>> results = reranker.rerank(query, documents)
    """

    # Popular BGE reranker models
    MODELS = {
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    }

    def __init__(self, config: RerankConfig):
        """
        Initialize BGE reranker.

        Args:
            config: Reranking configuration

        Raises:
            ImportError: If transformers is not installed
        """
        super().__init__(config)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for BGEReranker. "
                "Install it with: pip install transformers torch"
            )

        # Get model name
        model_name = config.model_name
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        # Load model and tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BGE reranker model: {model_name} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.model_name = model_name

        logger.info(f"BGE reranker model loaded successfully: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents using BGE reranker.

        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores (not used by BGE)

        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance

        Raises:
            ValueError: If inputs are invalid
        """
        self.validate_inputs(query, documents)

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Process in batches
        batch_size = self.config.batch_size
        all_scores = []

        logger.debug(f"Re-ranking {len(documents)} documents in batches of {batch_size}")

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get scores
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Extract scores (BGE models output a single score per pair)
                batch_scores = logits.squeeze(-1).cpu().tolist()

                # Handle single item case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                all_scores.extend(batch_scores)

        # Create (index, score) pairs and sort by score descending
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranking complete. Top score: {indexed_scores[0][1]:.4f}")

        return indexed_scores

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
