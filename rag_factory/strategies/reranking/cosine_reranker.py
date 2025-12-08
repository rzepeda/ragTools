"""
Cosine similarity reranker implementation.

This module provides a lightweight reranker using cosine similarity
with numpy for vector operations. No PyTorch dependencies required.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import logging

from .base import IReranker, RerankConfig

logger = logging.getLogger(__name__)


class CosineReranker(IReranker):
    """
    Lightweight re-ranker using cosine similarity.
    
    Uses embedding provider to generate query and document embeddings,
    then ranks documents by similarity to the query. Supports multiple
    similarity metrics and requires no external ML frameworks.
    
    Example:
        >>> from rag_factory.services.embedding import ONNXEmbeddingProvider
        >>> embedder = ONNXEmbeddingProvider()
        >>> config = RerankConfig(model_name="cosine")
        >>> reranker = CosineReranker(config, embedding_provider=embedder)
        >>> results = reranker.rerank(query, documents)
    """
    
    SUPPORTED_METRICS = ["cosine", "dot", "euclidean"]
    
    def __init__(
        self,
        config: RerankConfig,
        embedding_provider: Any = None,
        metric: str = "cosine",
        normalize: bool = True
    ):
        """
        Initialize cosine similarity reranker.
        
        Args:
            config: Reranking configuration
            embedding_provider: Provider for generating embeddings
            metric: Similarity metric ("cosine", "dot", "euclidean")
            normalize: Whether to normalize embeddings before comparison
            
        Raises:
            ValueError: If metric is not supported or embedding_provider is None
        """
        super().__init__(config)
        
        if embedding_provider is None:
            raise ValueError(
                "CosineReranker requires an embedding_provider. "
                "Pass an embedding provider instance (e.g., ONNXEmbeddingProvider)"
            )
        
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Supported metrics: {', '.join(self.SUPPORTED_METRICS)}"
            )
        
        self.embedding_provider = embedding_provider
        self.metric = metric
        self.normalize = normalize
        
        logger.info(
            f"Initialized CosineReranker with metric: {metric}, "
            f"normalize: {normalize}"
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents using cosine similarity.
        
        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores (not used by cosine reranker)
            
        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance
            
        Raises:
            ValueError: If inputs are invalid
        """
        self.validate_inputs(query, documents)
        
        logger.debug(
            f"Re-ranking {len(documents)} documents using {self.metric} similarity"
        )
        
        # Embed query
        query_embedding = np.array(
            self.embedding_provider.embed_query(query),
            dtype=np.float32
        )
        
        # Embed documents
        doc_embeddings = np.array(
            self.embedding_provider.embed_documents(documents),
            dtype=np.float32
        )
        
        # Normalize if requested
        if self.normalize:
            query_embedding = self._normalize_vector(query_embedding)
            doc_embeddings = self._normalize_batch(doc_embeddings)
        
        # Calculate similarities based on metric
        if self.metric == "cosine":
            similarities = self._cosine_similarity(query_embedding, doc_embeddings)
        elif self.metric == "dot":
            similarities = self._dot_product(query_embedding, doc_embeddings)
        elif self.metric == "euclidean":
            similarities = self._euclidean_similarity(query_embedding, doc_embeddings)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Create (index, score) pairs and sort by score descending
        indexed_scores = list(enumerate(similarities.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(
            f"Re-ranking complete. Top score: {indexed_scores[0][1]:.4f}, "
            f"Bottom score: {indexed_scores[-1][1]:.4f}"
        )
        
        return indexed_scores
    
    def _cosine_similarity(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.
        
        For normalized vectors, cosine similarity equals dot product.
        
        Args:
            query: Query embedding vector (1D)
            documents: Document embedding matrix (2D)
            
        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # For normalized vectors, cosine = dot product
            return np.dot(documents, query)
        else:
            # Calculate cosine similarity manually
            query_norm = np.linalg.norm(query)
            doc_norms = np.linalg.norm(documents, axis=1)
            
            # Avoid division by zero
            query_norm = max(query_norm, 1e-9)
            doc_norms = np.clip(doc_norms, a_min=1e-9, a_max=None)
            
            dot_products = np.dot(documents, query)
            return dot_products / (query_norm * doc_norms)
    
    def _dot_product(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """
        Calculate dot product between query and documents.
        
        Args:
            query: Query embedding vector (1D)
            documents: Document embedding matrix (2D)
            
        Returns:
            Array of dot product scores
        """
        return np.dot(documents, query)
    
    def _euclidean_similarity(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """
        Calculate euclidean similarity (inverse distance).
        
        Converts euclidean distance to similarity score where
        smaller distance = higher similarity.
        
        Args:
            query: Query embedding vector (1D)
            documents: Document embedding matrix (2D)
            
        Returns:
            Array of similarity scores
        """
        # Calculate euclidean distances
        distances = np.linalg.norm(documents - query, axis=1)
        
        # Convert distance to similarity (smaller distance = higher similarity)
        # Using 1 / (1 + distance) to map to [0, 1] range
        similarities = 1.0 / (1.0 + distances)
        
        return similarities
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a single vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return vector
        return vector / norm
    
    def _normalize_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of vectors to unit length.
        
        Args:
            vectors: Input matrix of vectors (2D)
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return vectors / norms
    
    def get_model_name(self) -> str:
        """Get the reranker model name."""
        return f"cosine-{self.metric}"
