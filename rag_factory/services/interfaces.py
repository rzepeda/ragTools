"""Service interfaces for dependency injection.

This module defines abstract base classes for all external services
that RAG strategies depend on. These interfaces enable dependency injection
and allow strategies to depend on contracts rather than implementations.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any, Tuple


class ILLMService(ABC):
    """Interface for Large Language Model services.
    
    This interface defines the contract for LLM services that generate
    text completions. Implementations can use different providers
    (OpenAI, Anthropic, local models, etc.) as long as they conform
    to this interface.
    
    Example:
        >>> class MyLLMService(ILLMService):
        ...     async def complete(self, prompt: str, **kwargs) -> str:
        ...         return "Generated response"
        ...     async def stream_complete(self, prompt: str, **kwargs):
        ...         yield "Generated"
        ...         yield " response"
        >>> service = MyLLMService()
        >>> response = await service.complete("Hello")
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate completion for prompt.

        Args:
            prompt: Input text prompt for generation
            max_tokens: Maximum number of tokens to generate (provider-specific default if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text completion

        Raises:
            Exception: If completion generation fails
        """

    @abstractmethod
    async def stream_complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            prompt: Input text prompt for generation
            max_tokens: Maximum number of tokens to generate (provider-specific default if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional provider-specific parameters

        Yields:
            Generated text tokens as they are produced

        Raises:
            Exception: If streaming fails
        """


class IEmbeddingService(ABC):
    """Interface for embedding generation services.
    
    This interface defines the contract for services that convert text
    into vector embeddings. Implementations can use different models
    (OpenAI, Cohere, local models, etc.) as long as they conform to
    this interface.
    
    Example:
        >>> class MyEmbeddingService(IEmbeddingService):
        ...     async def embed(self, text: str) -> List[float]:
        ...         return [0.1, 0.2, 0.3]
        ...     async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...         return [[0.1, 0.2, 0.3] for _ in texts]
        ...     def get_dimension(self) -> int:
        ...         return 3
        >>> service = MyEmbeddingService()
        >>> embedding = await service.embed("Hello world")
    """

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        This method is more efficient than calling embed() multiple times
        as it can batch requests to the underlying provider.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            Exception: If embedding generation fails
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Number of dimensions in the embedding vectors
        """


class IGraphService(ABC):
    """Interface for graph database services.
    
    This interface defines the contract for graph database operations
    used in knowledge graph RAG strategies. Implementations can use
    different backends (Neo4j, in-memory, etc.) as long as they conform
    to this interface.
    
    Example:
        >>> class MyGraphService(IGraphService):
        ...     async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        ...         return "node_123"
        ...     async def create_relationship(self, from_node_id: str, to_node_id: str,
        ...                                   relationship_type: str, properties=None):
        ...         pass
        ...     async def query(self, cypher_query: str, parameters=None) -> List[Dict[str, Any]]:
        ...         return []
        >>> service = MyGraphService()
        >>> node_id = await service.create_node("Person", {"name": "Alice"})
    """

    @abstractmethod
    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in the graph.

        Args:
            label: Node label/type (e.g., "Entity", "Document", "Person")
            properties: Dictionary of node properties

        Returns:
            Unique identifier for the created node

        Raises:
            Exception: If node creation fails
        """
    @abstractmethod
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create relationship between nodes.

        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            relationship_type: Type of relationship (e.g., "RELATES_TO", "MENTIONS")
            properties: Optional dictionary of relationship properties

        Raises:
            Exception: If relationship creation fails or nodes don't exist
        """
    @abstractmethod
    async def query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query.

        Args:
            cypher_query: Cypher query string to execute
            parameters: Optional parameters for the query (for parameterized queries)

        Returns:
            List of result records as dictionaries

        Raises:
            Exception: If query execution fails or query is invalid
        """


class IRerankingService(ABC):
    """Interface for document reranking services.
    
    This interface defines the contract for services that rerank documents
    by relevance to a query. Implementations can use different models
    (Cohere, cross-encoders, etc.) as long as they conform to this interface.
    
    Example:
        >>> class MyRerankingService(IRerankingService):
        ...     async def rerank(self, query: str, documents: List[str],
        ...                     top_k: int = 5) -> List[Tuple[int, float]]:
        ...         return [(0, 0.95), (2, 0.87), (1, 0.76)]
        >>> service = MyRerankingService()
        >>> results = await service.rerank("query", ["doc1", "doc2", "doc3"])
    """

    @abstractmethod
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


class IDatabaseService(ABC):
    """Interface for database operations.
    
    This interface defines the contract for database services that store
    and retrieve document chunks with vector embeddings. Implementations
    can use different backends (PostgreSQL with pgvector, Qdrant, etc.)
    as long as they conform to this interface.
    
    Example:
        >>> class MyDatabaseService(IDatabaseService):
        ...     async def store_chunks(self, chunks: List[Dict[str, Any]]):
        ...         pass
        ...     async def search_chunks(self, query_embedding: List[float],
        ...                           top_k: int = 10) -> List[Dict[str, Any]]:
        ...         return []
        ...     async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        ...         return {"id": chunk_id, "text": "content"}
        >>> service = MyDatabaseService()
        >>> await service.store_chunks([{"text": "Hello", "embedding": [0.1, 0.2]}])
    """

    @abstractmethod
    async def store_chunks(self, chunks: List[Dict[str, Any]], table_name: Optional[str] = None) -> None:
        """Store document chunks.

        Args:
            chunks: List of chunk dictionaries. Each chunk should contain
                   at minimum: text content and embedding vector. Additional
                   fields like metadata, chunk_id, etc. are implementation-specific.

        Raises:
            Exception: If storage fails
        """
    @abstractmethod
    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chunks by similarity.

        Performs vector similarity search to find chunks most similar
        to the query embedding.

        Args:
            query_embedding: Query vector to search for
            top_k: Maximum number of results to return

        Returns:
            List of chunk dictionaries, sorted by similarity score in
            descending order. Each chunk should contain at minimum the
            text content and similarity score.

        Raises:
            Exception: If search fails
        """
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve chunk by ID.

        Args:
            chunk_id: Unique identifier of the chunk to retrieve

        Returns:
            Chunk dictionary containing text content and metadata

        Raises:
            Exception: If chunk is not found or retrieval fails
        """
    @abstractmethod
    async def get_chunks_for_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a list of documents.

        Args:
            document_ids: List of document IDs to retrieve chunks for

        Returns:
            List of chunk dictionaries belonging to the specified documents
        """
    
    @abstractmethod
    async def store_chunks_with_hierarchy(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks with hierarchical metadata.
        
        This method stores chunks that include hierarchical relationship metadata
        such as parent-child relationships, hierarchy levels, and path information.
        
        Args:
            chunks: List of chunk dictionaries with hierarchy metadata.
                   Each chunk should contain:
                   - id: Unique chunk identifier
                   - document_id: ID of the parent document
                   - text: Chunk text content
                   - level: Hierarchy level (0 = document, 1 = section, 2 = paragraph)
                   - parent_id: ID of parent chunk (None for root level)
                   - path: List of indices representing path from root
                   - metadata: Additional metadata dictionary
        
        Raises:
            Exception: If storage fails
        """

