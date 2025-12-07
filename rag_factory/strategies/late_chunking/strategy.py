"""
Late chunking RAG strategy.

This module implements the main late chunking strategy that embeds documents
before chunking to maintain full context.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

from .document_embedder import DocumentEmbedder
from .embedding_chunker import EmbeddingChunker
from .coherence_analyzer import CoherenceAnalyzer
from .models import LateChunkingConfig, EmbeddingChunkingMethod

logger = logging.getLogger(__name__)


class LateChunkingRAGStrategy:
    """
    Late Chunking RAG: Embed first, then chunk.

    This experimental strategy embeds the full document before chunking,
    maintaining full context during embedding for potentially better
    semantic understanding.
    """

    def __init__(
        self,
        vector_store_service: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize late chunking strategy.

        Args:
            vector_store_service: Vector store for indexing chunks
            config: Configuration dictionary
        """
        # Create config object
        if config:
            # Convert string chunking_method to enum if needed
            if "chunking_method" in config and isinstance(config["chunking_method"], str):
                config["chunking_method"] = EmbeddingChunkingMethod(config["chunking_method"])
            self.late_config = LateChunkingConfig(**config)
        else:
            self.late_config = LateChunkingConfig()

        # Initialize components
        self.vector_store = vector_store_service
        self.document_embedder = DocumentEmbedder(self.late_config)
        self.embedding_chunker = EmbeddingChunker(self.late_config)
        self.coherence_analyzer = CoherenceAnalyzer(self.late_config)

        logger.info("Late Chunking RAG Strategy initialized")
        logger.info(f"Model: {self.late_config.model_name}")
        logger.info(f"Chunking method: {self.late_config.chunking_method.value}")

    def index_document(self, document: str, document_id: str) -> None:
        """
        Index document using late chunking.

        Args:
            document: Document text
            document_id: Unique document ID
        """
        logger.info(f"Indexing document with late chunking: {document_id}")

        # Step 1: Embed full document
        doc_embedding = self.document_embedder.embed_document(document, document_id)
        logger.info(
            f"Document embedded: {doc_embedding.token_count} tokens, "
            f"dim {doc_embedding.embedding_dim}"
        )

        # Step 2: Chunk embeddings
        chunks = self.embedding_chunker.chunk_embeddings(doc_embedding)
        logger.info(f"Created {len(chunks)} embedding chunks")

        # Step 3: Analyze coherence
        if self.late_config.compute_coherence_scores:
            chunks = self.coherence_analyzer.analyze_chunk_coherence(chunks)
            coherence_scores = [c.coherence_score for c in chunks if c.coherence_score]
            if coherence_scores:
                avg_coherence = np.mean(coherence_scores)
                logger.info(f"Average coherence score: {avg_coherence:.3f}")

        # Step 4: Index chunks in vector store
        for chunk in chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=chunk.chunk_embedding,
                metadata={
                    "document_id": chunk.document_id,
                    "token_count": chunk.token_count,
                    "coherence_score": chunk.coherence_score,
                    "chunking_method": "late_chunking",
                    "token_range": chunk.token_range,
                    "char_range": chunk.char_range
                }
            )

        logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using late chunking embeddings.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            List of retrieved chunks
        """
        logger.info(f"Late chunking retrieval for: {query}")

        # Use vector store search
        results = self.vector_store.search(query, top_k=top_k, **kwargs)

        # Add late chunking metadata to results
        for result in results:
            result["strategy"] = "late_chunking"

        return results

    @property
    def name(self) -> str:
        """Strategy name."""
        return "late_chunking"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Embed full document first, then chunk embeddings for better context"
