"""
Cohere reranker implementation.

This module provides integration with Cohere's Rerank API for state-of-the-art
reranking with minimal setup.
"""

from typing import List, Tuple, Optional
import logging

try:
    import cohere
    from tenacity import retry, stop_after_attempt, wait_exponential
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from .base import IReranker, RerankConfig

logger = logging.getLogger(__name__)


class CohereReranker(IReranker):
    """
    Re-ranker using Cohere's Rerank API.

    Provides state-of-the-art re-ranking with minimal setup using Cohere's
    managed service.

    Example:
        >>> config = RerankConfig(
        ...     model_name="rerank-english-v2.0",
        ...     model_config={"api_key": "your-api-key"}
        ... )
        >>> reranker = CohereReranker(config)
        >>> results = reranker.rerank(query, documents)
    """

    MODELS = {
        "rerank-english-v2.0": "rerank-english-v2.0",
        "rerank-multilingual-v2.0": "rerank-multilingual-v2.0"
    }

    def __init__(self, config: RerankConfig):
        """
        Initialize Cohere reranker.

        Args:
            config: Reranking configuration with api_key in model_config

        Raises:
            ImportError: If cohere package is not installed
            ValueError: If API key is not provided
        """
        super().__init__(config)

        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install it with: pip install cohere tenacity"
            )

        # Get API key from config
        api_key = config.model_config.get("api_key")
        if not api_key:
            raise ValueError("Cohere API key required in model_config['api_key']")

        self.client = cohere.Client(api_key)
        self.model_name = config.model_name or "rerank-english-v2.0"

        logger.info(f"Cohere reranker initialized with model: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents using Cohere Rerank API.

        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores (not used by Cohere)

        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance

        Raises:
            ValueError: If inputs are invalid
        """
        self.validate_inputs(query, documents)

        logger.debug(f"Calling Cohere Rerank API for {len(documents)} documents")

        # Call Cohere Rerank API
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=len(documents),  # Return all, we'll filter later
            return_documents=False
        )

        # Extract results
        results = []
        for result in response.results:
            results.append((result.index, result.relevance_score))

        logger.debug(f"Cohere rerank complete. Top score: {results[0][1]:.4f}")

        # Results are already sorted by relevance
        return results

    def get_model_name(self) -> str:
        """Get the model name."""
        return f"cohere:{self.model_name}"
