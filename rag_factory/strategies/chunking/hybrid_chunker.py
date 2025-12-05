"""Hybrid chunking strategy combining structural and semantic approaches."""

from typing import List, Dict, Any, Optional
import logging

from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod
from .structural_chunker import StructuralChunker
from .semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)


class HybridChunker(IChunker):
    """Hybrid chunking combining structural and semantic approaches.

    First uses structural chunking to split by document structure (headers, paragraphs),
    then applies semantic chunking within large sections to find optimal boundaries.

    This provides the best of both worlds: respects document structure while
    ensuring semantic coherence.

    Attributes:
        config: Chunking configuration
        structural_chunker: Structural chunker instance
        semantic_chunker: Optional semantic chunker instance
        use_semantic: Whether to use semantic refinement
    """

    def __init__(self, config: ChunkingConfig, embedding_service=None):
        """Initialize hybrid chunker.

        Args:
            config: Chunking configuration
            embedding_service: Optional embedding service for semantic chunking
        """
        super().__init__(config)

        # Always create structural chunker
        self.structural_chunker = StructuralChunker(config)

        # Create semantic chunker if embeddings enabled and service provided
        self.use_semantic = config.use_embeddings and embedding_service is not None

        if self.use_semantic:
            self.semantic_chunker = SemanticChunker(config, embedding_service)
        else:
            self.semantic_chunker = None
            logger.info(
                "Semantic chunking disabled. Using structural chunking only. "
                "Provide embedding_service to enable semantic refinement."
            )

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document using hybrid approach.

        Args:
            document: Document text to chunk
            document_id: Unique document identifier

        Returns:
            List of Chunk objects
        """
        if not document or not document.strip():
            return []

        # Step 1: Use structural chunking to get initial chunks
        structural_chunks = self.structural_chunker.chunk_document(document, document_id)

        if not structural_chunks:
            return []

        # Step 2: If semantic chunking disabled, return structural chunks
        if not self.use_semantic or not self.semantic_chunker:
            # Update chunking method to HYBRID
            for chunk in structural_chunks:
                chunk.metadata.chunking_method = ChunkingMethod.HYBRID
            return structural_chunks

        # Step 3: Apply semantic chunking to large chunks
        refined_chunks = []

        for chunk in structural_chunks:
            # If chunk is within acceptable size, keep as is
            if chunk.metadata.token_count <= self.config.target_chunk_size:
                chunk.metadata.chunking_method = ChunkingMethod.HYBRID
                refined_chunks.append(chunk)
                continue

            # If chunk is too large and not atomic, apply semantic chunking
            if not self._is_atomic_content(chunk.text):
                try:
                    # Use semantic chunker to split this chunk
                    semantic_sub_chunks = self.semantic_chunker.chunk_document(
                        chunk.text,
                        chunk.metadata.chunk_id
                    )

                    # Update metadata to reflect hybrid approach and hierarchy
                    for sub_chunk in semantic_sub_chunks:
                        sub_chunk.metadata.chunking_method = ChunkingMethod.HYBRID
                        sub_chunk.metadata.source_document_id = document_id
                        sub_chunk.metadata.section_hierarchy = chunk.metadata.section_hierarchy
                        sub_chunk.metadata.parent_chunk_id = chunk.metadata.chunk_id

                    refined_chunks.extend(semantic_sub_chunks)

                except Exception as e:
                    logger.warning(
                        f"Semantic chunking failed for chunk {chunk.metadata.chunk_id}: {e}. "
                        "Keeping original chunk."
                    )
                    chunk.metadata.chunking_method = ChunkingMethod.HYBRID
                    refined_chunks.append(chunk)
            else:
                # Keep atomic content as is
                chunk.metadata.chunking_method = ChunkingMethod.HYBRID
                refined_chunks.append(chunk)

        # Step 4: Reindex chunks
        for i, chunk in enumerate(refined_chunks):
            chunk.metadata.position = i
            # Update chunk_id to reflect final position
            chunk.metadata.chunk_id = f"{document_id}_chunk_{i}"

        # Step 5: Compute coherence scores if configured and semantic chunking enabled
        if self.config.compute_coherence_scores and self.semantic_chunker:
            try:
                refined_chunks = self._compute_coherence_scores(refined_chunks)
            except Exception as e:
                logger.warning(f"Failed to compute coherence scores: {e}")

        return refined_chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents.

        Args:
            documents: List of dicts with 'text' and 'id' keys

        Returns:
            List of chunk lists, one per document
        """
        return [
            self.chunk_document(doc["text"], doc["id"])
            for doc in documents
        ]

    def _compute_coherence_scores(self, chunks: List[Chunk]) -> List[Chunk]:
        """Compute coherence scores using semantic chunker.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with coherence scores
        """
        if not self.semantic_chunker:
            return chunks

        return self.semantic_chunker._compute_coherence_scores(chunks)

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunk distribution.

        Extends base stats with hybrid-specific information.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with statistics
        """
        stats = super().get_stats(chunks)

        if chunks:
            # Count chunks with hierarchy (from structural chunking)
            chunks_with_hierarchy = sum(
                1 for c in chunks
                if c.metadata.section_hierarchy
            )

            # Count chunks with parent (semantically split)
            chunks_with_parent = sum(
                1 for c in chunks
                if c.metadata.parent_chunk_id
            )

            stats["chunks_with_hierarchy"] = chunks_with_hierarchy
            stats["chunks_semantically_split"] = chunks_with_parent
            stats["semantic_refinement_enabled"] = self.use_semantic

        return stats
