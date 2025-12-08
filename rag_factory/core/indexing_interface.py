"""Indexing strategy interface and context.

This module defines the interface for document indexing strategies and
the shared context used during indexing operations. It enables separation
of indexing from retrieval pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, TYPE_CHECKING

from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies

if TYPE_CHECKING:
    from rag_factory.services.interfaces import IDatabaseService


class IndexingContext:
    """Shared context for indexing operations.

    This class holds shared resources and configuration that are used
    across indexing operations. It provides a centralized place for:
    - Database service access
    - Configuration parameters
    - Performance metrics tracking

    Attributes:
        database: Database service for storing indexed data
        config: Configuration dictionary for indexing parameters
        metrics: Dictionary for tracking performance metrics

    Example:
        >>> from unittest.mock import Mock
        >>> db_service = Mock()
        >>> context = IndexingContext(
        ...     database_service=db_service,
        ...     config={"chunk_size": 512}
        ... )
        >>> context.metrics["chunks_processed"] = 100
        >>> context.config["chunk_size"]
        512
    """

    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: Dict[str, Any] = None
    ):
        """Initialize indexing context.

        Args:
            database_service: Database service for storing indexed data
            config: Optional configuration dictionary
        """
        self.database = database_service
        self.config = config or {}
        self.metrics: Dict[str, Any] = {}


class IIndexingStrategy(ABC):
    """Interface for document indexing strategies.

    This abstract base class defines the contract that all indexing
    strategies must implement. It enforces:
    - Dependency declaration and validation
    - Capability declaration (what the strategy produces)
    - Async document processing

    Indexing strategies are responsible for processing documents and
    creating searchable indices (vectors, keywords, graphs, etc.) that
    can later be used by retrieval strategies.

    Example:
        >>> class MyIndexingStrategy(IIndexingStrategy):
        ...     def produces(self) -> Set[IndexCapability]:
        ...         return {IndexCapability.VECTORS, IndexCapability.DATABASE}
        ...
        ...     def requires_services(self) -> Set[ServiceDependency]:
        ...         return {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
        ...
        ...     async def process(self, documents, context):
        ...         # Implementation here
        ...         return IndexingResult(
        ...             capabilities=self.produces(),
        ...             metadata={},
        ...             document_count=len(documents),
        ...             chunk_count=0
        ...         )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """Initialize indexing strategy.

        Validates that all required services are present in the
        dependencies container. Raises an error if any required
        services are missing.

        Args:
            config: Strategy-specific configuration
            dependencies: Injected services (validated at creation)

        Raises:
            ValueError: If required services are missing

        Example:
            >>> from unittest.mock import Mock
            >>> deps = StrategyDependencies(
            ...     embedding_service=Mock(),
            ...     database_service=Mock()
            ... )
            >>> config = {"chunk_size": 512}
            >>> # This would work for a strategy requiring EMBEDDING and DATABASE
        """
        self.config = config
        self.deps = dependencies

        # Validate dependencies
        from rag_factory.services.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())

    @abstractmethod
    def produces(self) -> Set[IndexCapability]:
        """Declare what capabilities this strategy produces.

        This method must return the set of capabilities that will be
        available after this indexing strategy completes. These capabilities
        can then be used by retrieval strategies.

        Returns:
            Set of IndexCapability enums representing what this strategy produces

        Example:
            >>> def produces(self) -> Set[IndexCapability]:
            ...     return {
            ...         IndexCapability.VECTORS,
            ...         IndexCapability.CHUNKS,
            ...         IndexCapability.DATABASE
            ...     }
        """
        pass

    @abstractmethod
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare what services this strategy requires.

        This method must return the set of services that this strategy
        needs to function. The framework will validate that all required
        services are available before allowing the strategy to execute.

        Returns:
            Set of ServiceDependency enums representing required services

        Example:
            >>> def requires_services(self) -> Set[ServiceDependency]:
            ...     return {
            ...         ServiceDependency.EMBEDDING,
            ...         ServiceDependency.DATABASE
            ...     }
        """
        pass

    @abstractmethod
    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Process documents for indexing.

        This method performs the actual indexing work. It should:
        1. Process the input documents
        2. Create searchable indices (vectors, keywords, etc.)
        3. Store the indices using the context's database service
        4. Return a result describing what was produced

        Args:
            documents: List of document dictionaries to index.
                      Each document should have at minimum a 'text' field.
            context: Shared indexing context with database service and config

        Returns:
            IndexingResult with capabilities produced and metadata

        Example:
            >>> async def process(self, documents, context):
            ...     # Chunk documents
            ...     chunks = self._chunk_documents(documents)
            ...
            ...     # Generate embeddings
            ...     embeddings = await self.deps.embedding_service.embed_batch(
            ...         [c['text'] for c in chunks]
            ...     )
            ...
            ...     # Store in database
            ...     await context.database.store_chunks(chunks)
            ...
            ...     return IndexingResult(
            ...         capabilities=self.produces(),
            ...         metadata={"duration": 1.5},
            ...         document_count=len(documents),
            ...         chunk_count=len(chunks)
            ...     )
        """
        pass


class VectorEmbeddingIndexing(IIndexingStrategy):
    """Example indexing strategy using vector embeddings.

    This is a reference implementation that demonstrates how to:
    - Declare capabilities and service requirements
    - Validate dependencies
    - Use injected services
    - Process documents and create indices
    - Use the indexing context

    This strategy:
    - Chunks documents into smaller pieces
    - Generates vector embeddings for each chunk
    - Stores chunks and embeddings in the database

    Produces:
        - VECTORS: Vector embeddings for semantic search
        - CHUNKS: Document chunks for retrieval
        - DATABASE: Data persisted to database

    Requires:
        - EMBEDDING: Service for generating embeddings
        - DATABASE: Service for storing chunks

    Example:
        >>> from unittest.mock import Mock, AsyncMock
        >>> embedding_service = Mock()
        >>> embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        >>> database_service = Mock()
        >>> database_service.store_chunks = AsyncMock()
        >>>
        >>> deps = StrategyDependencies(
        ...     embedding_service=embedding_service,
        ...     database_service=database_service
        ... )
        >>> config = {"chunk_size": 512, "chunk_overlap": 50}
        >>> strategy = VectorEmbeddingIndexing(config, deps)
        >>>
        >>> context = IndexingContext(database_service, config)
        >>> documents = [{"text": "Sample document text"}]
        >>> # result = await strategy.process(documents, context)
    """

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.

        Returns:
            Set containing VECTORS, CHUNKS, and DATABASE capabilities
        """
        return {
            IndexCapability.VECTORS,
            IndexCapability.CHUNKS,
            IndexCapability.DATABASE
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Set containing EMBEDDING and DATABASE service dependencies
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
        """Process documents by chunking and embedding.

        Args:
            documents: List of document dictionaries with 'text' field
            context: Indexing context with database service

        Returns:
            IndexingResult with capabilities and metrics
        """
        # Get configuration
        chunk_size = self.config.get("chunk_size", 512)
        chunk_overlap = self.config.get("chunk_overlap", 50)

        # Chunk documents
        chunks = self._chunk_documents(documents, chunk_size, chunk_overlap)

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.deps.embedding_service.embed_batch(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        # Store in database
        await context.database.store_chunks(chunks)

        # Update metrics
        context.metrics["chunks_created"] = len(chunks)
        context.metrics["documents_processed"] = len(documents)

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )

    def _chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Split documents into chunks.

        Simple character-based chunking for demonstration purposes.

        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")

            # Simple character-based chunking
            start = 0
            chunk_idx = 0

            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "text": chunk_text,
                    "document_id": doc.get("id", f"doc_{doc_idx}"),
                    "chunk_index": chunk_idx,
                    "metadata": doc.get("metadata", {})
                })

                start += chunk_size - chunk_overlap
                chunk_idx += 1

        return chunks
