"""
Embedding chunker for late chunking strategy.

This module splits token embeddings into semantically coherent chunks.
"""

from typing import List
import logging
import numpy as np

from .models import (
    DocumentEmbedding,
    EmbeddingChunk,
    EmbeddingChunkingMethod,
    LateChunkingConfig,
    TokenEmbedding
)

logger = logging.getLogger(__name__)


class EmbeddingChunker:
    """Split token embeddings into semantically coherent chunks."""

    def __init__(self, config: LateChunkingConfig):
        self.config = config

    def chunk_embeddings(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """
        Chunk token embeddings into semantic units.

        Args:
            doc_embedding: Document embedding with token details

        Returns:
            List of embedding chunks
        """
        logger.info(
            f"Chunking embeddings for document {doc_embedding.document_id} "
            f"using method: {self.config.chunking_method.value}"
        )

        method = self.config.chunking_method

        if method == EmbeddingChunkingMethod.FIXED_SIZE:
            chunks = self._fixed_size_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.SEMANTIC_BOUNDARY:
            chunks = self._semantic_boundary_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.HIERARCHICAL:
            chunks = self._hierarchical_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.ADAPTIVE:
            chunks = self._adaptive_chunking(doc_embedding)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

        logger.info(f"Created {len(chunks)} embedding chunks")

        return chunks

    def _fixed_size_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Fixed-size chunking of token embeddings."""
        chunks = []
        tokens = doc_embedding.token_embeddings
        chunk_size = self.config.target_chunk_size
        overlap = self.config.chunk_overlap_tokens

        i = 0
        chunk_idx = 0

        while i < len(tokens):
            # Define chunk range
            start_idx = i
            end_idx = min(i + chunk_size, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Create chunk
            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                chunk_idx,
                EmbeddingChunkingMethod.FIXED_SIZE
            )

            chunks.append(chunk)

            # Move forward with overlap
            i += chunk_size - overlap
            chunk_idx += 1

        return chunks

    def _semantic_boundary_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Chunk based on semantic boundary detection in embeddings."""
        tokens = doc_embedding.token_embeddings

        # Calculate similarity between consecutive tokens
        boundaries = self._detect_semantic_boundaries(tokens)

        # Create chunks from boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_tokens = tokens[start_idx:end_idx]

            # Skip very small chunks
            if len(chunk_tokens) < self.config.min_chunk_size:
                # Merge with previous chunk if possible
                if chunks:
                    prev_chunk = chunks[-1]
                    # Update previous chunk to include these tokens
                    merged_tokens = tokens[boundaries[i-1]:end_idx]
                    chunks[-1] = self._create_chunk(
                        doc_embedding,
                        merged_tokens,
                        i - 1,
                        EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
                    )
                    continue
                # If this is the first chunk and it's small, keep it anyway
                # (better to have one small chunk than no chunks at all)
                elif i == 0 and len(boundaries) == 2:
                    # This is the only chunk (entire document is small)
                    chunk = self._create_chunk(
                        doc_embedding,
                        chunk_tokens,
                        i,
                        EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
                    )
                    chunks.append(chunk)
                    continue
                else:
                    continue

            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                i,
                EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
            )
            chunks.append(chunk)

        return chunks

    def _detect_semantic_boundaries(
        self,
        tokens: List[TokenEmbedding]
    ) -> List[int]:
        """Detect semantic boundaries by analyzing embedding similarities."""
        boundaries = [0]  # Start boundary

        # Extract embeddings as numpy array
        embeddings = np.array([t.embedding for t in tokens])

        # Calculate cosine similarity between consecutive tokens
        for i in range(1, len(embeddings) - 1):
            similarity = self._cosine_similarity(
                embeddings[i],
                embeddings[i + 1]
            )

            # If similarity drops below threshold, mark as boundary
            if similarity < self.config.similarity_threshold:
                # Check if chunk would be large enough
                if i - boundaries[-1] >= self.config.min_chunk_size:
                    boundaries.append(i)

        boundaries.append(len(tokens))  # End boundary

        return boundaries

    def _hierarchical_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Hierarchical chunking combining structure and embeddings."""
        # TODO: Implement hierarchical chunking
        # For now, fall back to semantic boundary
        logger.warning("Hierarchical chunking not fully implemented, using semantic boundary")
        return self._semantic_boundary_chunking(doc_embedding)

    def _adaptive_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Adaptive chunking based on embedding variance."""
        tokens = doc_embedding.token_embeddings
        embeddings = np.array([t.embedding for t in tokens])

        chunks = []
        current_chunk_start = 0
        chunk_idx = 0

        i = self.config.min_chunk_size

        while i < len(embeddings):
            # Calculate variance in current window
            window_embeddings = embeddings[current_chunk_start:i]
            variance = np.var(window_embeddings, axis=0).mean()

            # If variance is high, create chunk
            if variance > 0.1 and i - current_chunk_start >= self.config.min_chunk_size:
                chunk_tokens = tokens[current_chunk_start:i]
                chunk = self._create_chunk(
                    doc_embedding,
                    chunk_tokens,
                    chunk_idx,
                    EmbeddingChunkingMethod.ADAPTIVE
                )
                chunks.append(chunk)

                current_chunk_start = i
                chunk_idx += 1

            # Force chunk if max size reached
            if i - current_chunk_start >= self.config.max_chunk_size:
                chunk_tokens = tokens[current_chunk_start:i]
                chunk = self._create_chunk(
                    doc_embedding,
                    chunk_tokens,
                    chunk_idx,
                    EmbeddingChunkingMethod.ADAPTIVE
                )
                chunks.append(chunk)

                current_chunk_start = i
                chunk_idx += 1

            i += 1

        # Add remaining tokens
        if current_chunk_start < len(tokens):
            chunk_tokens = tokens[current_chunk_start:]
            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                chunk_idx,
                EmbeddingChunkingMethod.ADAPTIVE
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        doc_embedding: DocumentEmbedding,
        chunk_tokens: List[TokenEmbedding],
        chunk_idx: int,
        method: EmbeddingChunkingMethod
    ) -> EmbeddingChunk:
        """Create EmbeddingChunk from tokens."""
        # Extract text span
        start_char = chunk_tokens[0].start_char
        end_char = chunk_tokens[-1].end_char
        text = doc_embedding.text[start_char:end_char]

        # Average embeddings for chunk
        chunk_embedding = np.mean([t.embedding for t in chunk_tokens], axis=0).tolist()

        # Token range
        token_range = (chunk_tokens[0].position, chunk_tokens[-1].position)

        # Create chunk ID
        chunk_id = f"{doc_embedding.document_id}_late_chunk_{chunk_idx}"

        chunk = EmbeddingChunk(
            chunk_id=chunk_id,
            document_id=doc_embedding.document_id,
            text=text,
            chunk_embedding=chunk_embedding,
            token_range=token_range,
            char_range=(start_char, end_char),
            token_count=len(chunk_tokens),
            chunking_method=method
        )

        return chunk

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
