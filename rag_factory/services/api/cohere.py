"""Cohere Reranking service implementation.

This module provides a reranking service that implements IRerankingService
using Cohere's Rerank API.
"""

from typing import List, Tuple
import asyncio
import logging

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from rag_factory.services.interfaces import IRerankingService

logger = logging.getLogger(__name__)


class CohereRerankingService(IRerankingService):
    """Cohere Rerank API service.

    This service implements IRerankingService using Cohere's Rerank API
    for document reranking by relevance to a query.

    Example:
        >>> service = CohereRerankingService(api_key="your-api-key")
        >>> documents = ["Python is great", "Java is popular", "Python for ML"]
        >>> results = await service.rerank("Python programming", documents, top_k=2)
        >>> print(results)  # [(0, 0.95), (2, 0.87)]
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0"
    ):
        """Initialize Cohere reranking service.

        Args:
            api_key: Cohere API key
            model: Rerank model name (default: rerank-english-v3.0)

        Raises:
            ImportError: If cohere package is not installed
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere package not installed. Install with:\n"
                "  pip install cohere"
            )

        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key=api_key)

        logger.info(f"Initialized Cohere reranking service with model: {model}")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query to rank documents against
            documents: List of document texts to rerank
            top_k: Number of top documents to return

        Returns:
            List of (document_index, relevance_score) tuples, sorted by
            relevance score in descending order. Document indices refer
            to positions in the input documents list.

        Raises:
            Exception: If reranking fails
        """
        if not documents:
            return []

        try:
            # Run sync API call in executor to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=documents,
                    top_n=min(top_k, len(documents))
                )
            )

            # Extract results as (index, score) tuples
            results = [
                (result.index, result.relevance_score)
                for result in response.results
            ]

            logger.debug(
                f"Reranked {len(documents)} documents, "
                f"returning top {len(results)}"
            )

            return results

        except Exception as e:
            logger.error(f"Cohere reranking error: {e}")
            raise
