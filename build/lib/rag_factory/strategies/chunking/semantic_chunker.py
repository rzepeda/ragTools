"""Semantic chunking strategy using embedding-based boundary detection."""

from typing import List, Dict, Any, Optional
import re
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import numpy as np
except ImportError:
    np = None

from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod

logger = logging.getLogger(__name__)


class SemanticChunker(IChunker):
    """Chunks documents based on semantic similarity between segments.

    Uses embeddings to detect topic shifts and create semantically coherent chunks.
    This approach produces higher quality chunks at the cost of performance
    (requires embedding generation).

    Attributes:
        config: Chunking configuration
        embedding_service: Service for generating embeddings
        tokenizer: Tokenizer for counting tokens
    """

    def __init__(self, config: ChunkingConfig, embedding_service):
        """Initialize semantic chunker.

        Args:
            config: Chunking configuration
            embedding_service: EmbeddingService instance

        Raises:
            ImportError: If required dependencies not installed
        """
        super().__init__(config)

        if not config.use_embeddings:
            raise ValueError(
                "SemanticChunker requires use_embeddings=True in config"
            )

        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for SemanticChunker. "
                "Install with: pip install tiktoken"
            )

        if np is None:
            raise ImportError(
                "numpy is required for SemanticChunker. "
                "Install with: pip install numpy"
            )

        self.embedding_service = embedding_service

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback.")
            self.tokenizer = None

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document using semantic boundary detection.

        Args:
            document: Document text to chunk
            document_id: Unique document identifier

        Returns:
            List of Chunk objects
        """
        if not document or not document.strip():
            return []

        # Step 1: Split into sentences
        sentences = self._split_into_sentences(document)

        if not sentences:
            return []

        if len(sentences) == 1:
            # Single sentence - create one chunk
            return [self._create_single_chunk(sentences[0], document_id, 0)]

        # Step 2: Group sentences into segments for embedding
        segments = self._create_segments(sentences, segment_size=3)

        # Step 3: Generate embeddings for segments
        segment_texts = [" ".join(seg) for seg in segments]

        try:
            embedding_result = self.embedding_service.embed(segment_texts)
            embeddings = embedding_result.embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(document, document_id)

        # Step 4: Calculate similarity and detect boundaries
        boundaries = self._detect_boundaries(embeddings, len(sentences))

        # Step 5: Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, document_id
        )

        # Step 6: Adjust chunk sizes to meet constraints
        chunks = self._adjust_chunk_sizes(chunks, document_id)

        # Step 7: Compute coherence scores if configured
        if self.config.compute_coherence_scores and chunks:
            try:
                chunks = self._compute_coherence_scores(chunks)
            except Exception as e:
                logger.warning(f"Failed to compute coherence scores: {e}")

        return chunks

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

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # Splits on periods, exclamation marks, and question marks followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_segments(
        self,
        sentences: List[str],
        segment_size: int = 3
    ) -> List[List[str]]:
        """Group sentences into overlapping segments for embedding.

        Args:
            sentences: List of sentences
            segment_size: Number of sentences per segment

        Returns:
            List of sentence segments
        """
        segments = []
        for i in range(len(sentences)):
            segment = sentences[i:i + segment_size]
            if segment:
                segments.append(segment)
        return segments

    def _detect_boundaries(
        self,
        embeddings: List[List[float]],
        num_sentences: int
    ) -> List[int]:
        """Detect semantic boundaries based on embedding similarity.

        Args:
            embeddings: List of embedding vectors
            num_sentences: Total number of sentences

        Returns:
            List of indices where boundaries should be placed
        """
        if len(embeddings) < 2:
            return [0, num_sentences]

        boundaries = [0]  # Start with first sentence

        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # If similarity drops below threshold, it's a boundary
            if similarity < self.config.similarity_threshold:
                boundaries.append(i + 1)

        boundaries.append(num_sentences)  # End boundary
        return boundaries

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        document_id: str
    ) -> List[Chunk]:
        """Create chunks from detected boundaries.

        Args:
            sentences: List of sentences
            boundaries: List of boundary indices
            document_id: Document identifier

        Returns:
            List of chunks
        """
        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            if start_idx >= len(sentences):
                break

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if not chunk_text.strip():
                continue

            # Count tokens
            token_count = self._count_tokens(chunk_text)

            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                source_document_id=document_id,
                position=i,
                start_char=0,  # Would need to track actual character positions
                end_char=len(chunk_text),
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=token_count
            )

            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks

    def _adjust_chunk_sizes(
        self,
        chunks: List[Chunk],
        document_id: str
    ) -> List[Chunk]:
        """Adjust chunk sizes to meet min/max constraints.

        Args:
            chunks: List of chunks to adjust
            document_id: Document identifier

        Returns:
            Adjusted chunks
        """
        if not chunks:
            return chunks

        adjusted = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If chunk is too small, try to merge with next
            if current_chunk.metadata.token_count < self.config.min_chunk_size:
                if i + 1 < len(chunks):
                    # Merge with next chunk
                    next_chunk = chunks[i + 1]
                    merged_text = current_chunk.text + " " + next_chunk.text
                    merged_tokens = self._count_tokens(merged_text)

                    # Update metadata
                    current_chunk.text = merged_text
                    current_chunk.metadata.token_count = merged_tokens
                    current_chunk.metadata.end_char = next_chunk.metadata.end_char
                    i += 1  # Skip next chunk since we merged it

            # If chunk is too large, split it
            elif current_chunk.metadata.token_count > self.config.max_chunk_size:
                if not self._is_atomic_content(current_chunk.text):
                    # Use fixed-size splitting for oversized chunks
                    sub_chunks = self._split_large_chunk(current_chunk, document_id)
                    adjusted.extend(sub_chunks)
                    i += 1
                    continue

            adjusted.append(current_chunk)
            i += 1

        return adjusted

    def _split_large_chunk(self, chunk: Chunk, document_id: str) -> List[Chunk]:
        """Split a large chunk into smaller chunks.

        Args:
            chunk: Chunk to split
            document_id: Document identifier

        Returns:
            List of sub-chunks
        """
        sentences = self._split_into_sentences(chunk.text)
        sub_chunks = []
        current_text = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.config.target_chunk_size and current_text:
                # Create chunk from accumulated sentences
                text = " ".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{chunk.metadata.chunk_id}_sub_{len(sub_chunks)}",
                    source_document_id=chunk.metadata.source_document_id,
                    position=chunk.metadata.position,
                    start_char=0,
                    end_char=len(text),
                    section_hierarchy=chunk.metadata.section_hierarchy,
                    chunking_method=ChunkingMethod.SEMANTIC,
                    token_count=current_tokens,
                    parent_chunk_id=chunk.metadata.chunk_id
                )
                sub_chunks.append(Chunk(text=text, metadata=metadata))

                current_text = [sentence]
                current_tokens = sentence_tokens
            else:
                current_text.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining text
        if current_text:
            text = " ".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{chunk.metadata.chunk_id}_sub_{len(sub_chunks)}",
                source_document_id=chunk.metadata.source_document_id,
                position=chunk.metadata.position,
                start_char=0,
                end_char=len(text),
                section_hierarchy=chunk.metadata.section_hierarchy,
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=current_tokens,
                parent_chunk_id=chunk.metadata.chunk_id
            )
            sub_chunks.append(Chunk(text=text, metadata=metadata))

        return sub_chunks if sub_chunks else [chunk]

    def _compute_coherence_scores(self, chunks: List[Chunk]) -> List[Chunk]:
        """Compute intra-chunk coherence scores.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with coherence scores
        """
        for chunk in chunks:
            # Split chunk into sentences and compute internal similarity
            sentences = self._split_into_sentences(chunk.text)

            if len(sentences) <= 1:
                chunk.metadata.coherence_score = 1.0
                continue

            try:
                # Get embeddings for sentences
                result = self.embedding_service.embed(sentences)
                embeddings = result.embeddings

                # Calculate average pairwise similarity
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                    similarities.append(sim)

                chunk.metadata.coherence_score = (
                    sum(similarities) / len(similarities) if similarities else 1.0
                )
            except Exception as e:
                logger.warning(f"Failed to compute coherence for chunk: {e}")
                chunk.metadata.coherence_score = None

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to simple word count approximation
        return len(text.split())

    def _create_single_chunk(
        self,
        text: str,
        document_id: str,
        position: int
    ) -> Chunk:
        """Create a single chunk for short documents.

        Args:
            text: Chunk text
            document_id: Document identifier
            position: Chunk position

        Returns:
            Chunk object
        """
        token_count = self._count_tokens(text)

        metadata = ChunkMetadata(
            chunk_id=f"{document_id}_chunk_{position}",
            source_document_id=document_id,
            position=position,
            start_char=0,
            end_char=len(text),
            section_hierarchy=[],
            chunking_method=ChunkingMethod.SEMANTIC,
            token_count=token_count,
            coherence_score=1.0
        )

        return Chunk(text=text, metadata=metadata)

    def _fallback_chunking(self, document: str, document_id: str) -> List[Chunk]:
        """Fallback to simple chunking when semantic chunking fails.

        Args:
            document: Document text
            document_id: Document identifier

        Returns:
            List of chunks
        """
        logger.info("Using fallback chunking")

        sentences = self._split_into_sentences(document)
        if not sentences:
            return []

        chunks = []
        current_text = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.config.target_chunk_size and current_text:
                # Create chunk
                text = " ".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    source_document_id=document_id,
                    position=len(chunks),
                    start_char=0,
                    end_char=len(text),
                    section_hierarchy=[],
                    chunking_method=ChunkingMethod.SEMANTIC,
                    token_count=current_tokens
                )
                chunks.append(Chunk(text=text, metadata=metadata))

                current_text = [sentence]
                current_tokens = sentence_tokens
            else:
                current_text.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining text
        if current_text:
            text = " ".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                source_document_id=document_id,
                position=len(chunks),
                start_char=0,
                end_char=len(text),
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=current_tokens
            )
            chunks.append(Chunk(text=text, metadata=metadata))

        return chunks
