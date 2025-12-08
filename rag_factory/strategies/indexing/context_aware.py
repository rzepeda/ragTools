import logging
import re
from typing import List, Set, Dict, Any

import numpy as np

from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency

logger = logging.getLogger(__name__)


class ContextAwareChunkingIndexing(IIndexingStrategy):
    """Chunks documents at semantic boundaries using embeddings.

    This strategy splits documents into sentences, embeds them (or groups of them),
    and finds boundaries where the cosine similarity between consecutive embeddings
    drops below a threshold. It then merges sentences into chunks respecting these
    boundaries and configured size constraints.
    """

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.

        Returns:
            Set containing CHUNKS and DATABASE capabilities.
        """
        return {
            IndexCapability.CHUNKS,
            IndexCapability.DATABASE
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Set containing EMBEDDING and DATABASE service dependencies.
        """
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }

    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Process documents by chunking at semantic boundaries.

        Args:
            documents: List of document dictionaries to index.
            context: Shared indexing context.

        Returns:
            IndexingResult describing the outcome.
        """
        # Configuration
        chunk_size_min = self.config.get('chunk_size_min', 256)
        chunk_size_max = self.config.get('chunk_size_max', 1024)
        chunk_size_target = self.config.get('chunk_size_target', 512)
        boundary_threshold = self.config.get('boundary_threshold', 0.6)
        window_size = self.config.get('window_size', 3)

        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue

            doc_id = doc.get('id', 'unknown')

            # Step 1: Split into candidate sentences
            sentences = self._split_into_sentences(text)
            if not sentences:
                continue

            # Step 2: Create windows for embedding (context)
            windows = self._create_windows(sentences, window_size)

            # Step 3: Embed windows
            try:
                embeddings = await self.deps.embedding_service.embed_batch(windows)
            except Exception as e:
                logger.error("Failed to embed windows for document %s: %s", doc_id, e)
                raise

            # Step 4: Find semantic boundaries
            boundaries = self._find_boundaries(embeddings, boundary_threshold)

            # Step 5: Create chunks respecting boundaries and size constraints
            doc_chunks = self._create_chunks(
                sentences,
                boundaries,
                chunk_size_min,
                chunk_size_max
            )

            # Add metadata
            for i, chunk in enumerate(doc_chunks):
                chunk['metadata'] = chunk.get('metadata', {})
                chunk['metadata'].update({
                    'document_id': doc_id,
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks),
                    'strategy': 'context_aware'
                })
                # Preserve original document metadata
                if 'metadata' in doc:
                    chunk['metadata'].update(doc['metadata'])

            all_chunks.extend(doc_chunks)

        # Step 6: Store chunks
        if all_chunks:
            await context.database.store_chunks(all_chunks)

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'avg_chunk_size': (
                    sum(len(c['text']) for c in all_chunks) / len(all_chunks)
                    if all_chunks else 0
                ),
                'chunk_size_config': {
                    'min': chunk_size_min,
                    'max': chunk_size_max,
                    'target': chunk_size_target
                },
                'boundary_threshold': boundary_threshold
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple regex split for now, similar to SemanticChunker
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_windows(self, sentences: List[str], window_size: int) -> List[str]:
        """Create sliding windows of sentences for embedding."""
        windows = []
        if len(sentences) <= window_size:
            return [" ".join(sentences)]

        for i in range(len(sentences)):
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            window_text = " ".join(sentences[start:end])
            windows.append(window_text)

        return windows

    def _find_boundaries(
        self,
        embeddings: List[List[float]],
        threshold: float
    ) -> List[int]:
        """Find indices where semantic similarity drops below threshold.

        Returns a list of indices. An index `i` in the result means
        a boundary exists AFTER sentence `i` (i.e., between `i` and `i+1`).
        """
        boundaries = []
        # We compare embedding i with embedding i+1
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            if sim < threshold:
                boundaries.append(i)
        return boundaries

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _create_chunks(
        self,
        sentences: List[str],
        boundaries: List[int],
        min_size: int,
        max_size: int
    ) -> List[Dict[str, Any]]:
        """Merge sentences into chunks respecting boundaries and size constraints."""
        chunks = []
        current_chunk_sentences = []
        current_chunk_len = 0

        boundary_set = set(boundaries)

        for i, sentence in enumerate(sentences):
            sent_len = len(sentence)

            # If adding this sentence exceeds max size, we MUST split
            if current_chunk_sentences and (current_chunk_len + sent_len + 1 > max_size):
                chunks.append(self._finalize_chunk(current_chunk_sentences))
                current_chunk_sentences = []
                current_chunk_len = 0

            current_chunk_sentences.append(sentence)
            current_chunk_len += sent_len + 1  # +1 for space

            # Check if we are at a semantic boundary
            if i in boundary_set:
                if current_chunk_len >= min_size:
                    chunks.append(self._finalize_chunk(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_len = 0

        # Add remaining
        if current_chunk_sentences:
            chunks.append(self._finalize_chunk(current_chunk_sentences))

        return chunks

    def _finalize_chunk(self, sentences: List[str]) -> Dict[str, Any]:
        """Create a chunk dict from sentences."""
        text = " ".join(sentences)
        return {
            "text": text,
            # metadata will be added in process()
        }
