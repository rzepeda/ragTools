"""In-memory indexing strategy for testing.

This module implements an in-memory only indexing strategy that stores
chunks in a class-level dictionary. It's designed for unit tests that
don't require database setup.
"""

from typing import List, Dict, Any, Set, Optional

from rag_factory.core.indexing_interface import (
    IIndexingStrategy,
    IndexingContext,
    IndexingResult
)
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency


class InMemoryIndexing(IIndexingStrategy):
    """Stores chunks in memory for testing.

    This strategy provides fast, database-free indexing for unit tests.
    It uses a class-level dictionary to store chunks, allowing tests to
    verify indexing behavior without external dependencies.

    Produces:
        - CHUNKS: Document chunks created and stored
        - IN_MEMORY: Data stored in memory (not persisted)

    Requires:
        - No external services required

    Example:
        >>> from rag_factory.services.dependencies import StrategyDependencies
        >>> strategy = InMemoryIndexing(
        ...     config={"chunk_size": 512},
        ...     dependencies=StrategyDependencies()
        ... )
        >>> documents = [{"id": "doc1", "text": "Sample text"}]
        >>> # result = await strategy.process(documents, context)
        >>> chunk = InMemoryIndexing.get_chunk("doc1_chunk_0")
        >>> InMemoryIndexing.clear_storage()
    """

    # Class-level storage (shared across instances for testing)
    _storage: Dict[str, Dict[str, Any]] = {}

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.

        Returns:
            Set containing CHUNKS and IN_MEMORY capabilities
        """
        return {
            IndexCapability.CHUNKS,
            IndexCapability.IN_MEMORY
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Empty set (no services required)
        """
        return set()  # No services required!

    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Store chunks in memory for fast testing.

        This method chunks documents by character count and stores them
        in the class-level storage dictionary. Each chunk is assigned a
        unique ID based on the document ID and chunk index.

        Args:
            documents: List of documents to index. Each document should
                      have 'id' and 'text' fields.
            context: Indexing context (not used by this strategy)

        Returns:
            IndexingResult with capabilities and metrics

        Example:
            >>> documents = [
            ...     {"id": "doc1", "text": "Sample text for testing"},
            ...     {"id": "doc2", "text": "Another document"}
            ... ]
            >>> # result = await strategy.process(documents, context)
        """
        chunk_size = self.config.get('chunk_size', 512)
        all_chunks = []

        for doc in documents:
            doc_id = doc.get('id', f'doc_{id(doc)}')
            text = doc.get('text', '')

            # Simple chunking by character count
            chunks = self._chunk_by_size(text, chunk_size)

            for i, chunk_text in enumerate(chunks):
                chunk = {
                    'id': f"{doc_id}_chunk_{i}",
                    'document_id': doc_id,
                    'text': chunk_text,
                    'index': i,
                    'metadata': doc.get('metadata', {})
                }
                all_chunks.append(chunk)

        # Store in class-level dict
        for chunk in all_chunks:
            self._storage[chunk['id']] = chunk

        # Update context metrics
        context.metrics['chunks_created'] = len(all_chunks)
        context.metrics['documents_processed'] = len(documents)

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'storage_type': 'in_memory',
                'chunk_size': chunk_size,
                'total_chunks_stored': len(self._storage)
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )

    @classmethod
    def clear_storage(cls) -> None:
        """Clear in-memory storage (for test cleanup).

        This method should be called between tests to ensure a clean
        state and prevent test interference.

        Example:
            >>> InMemoryIndexing.clear_storage()
            >>> assert len(InMemoryIndexing.get_all_chunks()) == 0
        """
        cls._storage.clear()

    @classmethod
    def get_chunk(cls, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk from memory by ID.

        Args:
            chunk_id: The unique ID of the chunk to retrieve

        Returns:
            Chunk dictionary if found, None otherwise

        Example:
            >>> chunk = InMemoryIndexing.get_chunk("doc1_chunk_0")
            >>> if chunk:
            ...     print(chunk['text'])
        """
        return cls._storage.get(chunk_id)

    @classmethod
    def get_all_chunks(cls) -> List[Dict[str, Any]]:
        """Get all chunks from memory.

        Returns:
            List of all stored chunk dictionaries

        Example:
            >>> chunks = InMemoryIndexing.get_all_chunks()
            >>> print(f"Total chunks: {len(chunks)}")
        """
        return list(cls._storage.values())

    def _chunk_by_size(self, text: str, size: int) -> List[str]:
        """Simple chunking by character count.

        Splits text into chunks of approximately the specified size.
        This is a simple implementation for testing purposes.

        Args:
            text: The text to chunk
            size: Maximum size of each chunk in characters

        Returns:
            List of text chunks

        Example:
            >>> strategy = InMemoryIndexing({}, StrategyDependencies())
            >>> chunks = strategy._chunk_by_size("Hello World", 5)
            >>> len(chunks)
            3
        """
        if not text:
            return []

        chunks = []
        for i in range(0, len(text), size):
            chunks.append(text[i:i + size])
        return chunks
