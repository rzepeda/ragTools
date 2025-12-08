"""Cosine similarity reranking service implementation.

This module provides a reranking service that implements IRerankingService
using numpy-based cosine similarity for local reranking.
"""

from typing import List, Tuple
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from rag_factory.services.interfaces import IRerankingService, IEmbeddingService

logger = logging.getLogger(__name__)


class CosineRerankingService(IRerankingService):
    """Cosine similarity-based reranking service.

    This service implements IRerankingService using numpy for cosine
    similarity calculations. It requires an embedding service to convert
    text to vectors.

    Example:
        >>> embedding_service = ONNXEmbeddingService()
        >>> service = CosineRerankingService(embedding_service)
        >>> documents = ["Python is great", "Java is popular", "Python for ML"]
        >>> results = await service.rerank("Python programming", documents, top_k=2)
        >>> print(results)  # [(0, 0.95), (2, 0.87)]
    """

    def __init__(self, embedding_service: IEmbeddingService):
        """Initialize cosine reranking service.

        Args:
            embedding_service: Service to use for generating embeddings

        Raises:
            ImportError: If numpy is not installed
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy not installed. Install with:\n"
                "  pip install numpy"
            )

        self.embedding_service = embedding_service
        logger.info("Initialized cosine similarity reranking service")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Uses cosine similarity between query and document embeddings
        to determine relevance.

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
            # Generate embeddings for query and documents
            query_embedding = await self.embedding_service.embed(query)
            doc_embeddings = await self.embedding_service.embed_batch(documents)

            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)

            # Compute cosine similarities
            # cosine_similarity = dot(A, B) / (norm(A) * norm(B))
            # Since embeddings are typically normalized, we can just use dot product
            similarities = np.dot(doc_vecs, query_vec)

            # Create (index, score) tuples
            results = [
                (idx, float(score))
                for idx, score in enumerate(similarities)
            ]

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top_k results
            top_results = results[:min(top_k, len(results))]

            logger.debug(
                f"Reranked {len(documents)} documents using cosine similarity, "
                f"returning top {len(top_results)}"
            )

            return top_results

        except Exception as e:
            logger.error(f"Cosine reranking error: {e}")
            raise
