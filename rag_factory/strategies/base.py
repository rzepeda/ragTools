"""
Base interface for RAG (Retrieval-Augmented Generation) strategies.

This module defines the abstract base class and data structures that all RAG
strategies must implement. It provides a unified interface for different
retrieval and generation approaches.

Example usage:
    >>> class MyStrategy(IRAGStrategy):
    ...     def initialize(self, config: StrategyConfig) -> None:
    ...         self.config = config
    ...
    ...     def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
    ...         # Implementation here
    ...         pass
    ...
    ...     def retrieve(self, query: str, top_k: int) -> List[Chunk]:
    ...         # Implementation here
    ...         pass
    ...
    ...     async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
    ...         # Async implementation here
    ...         pass
    ...
    ...     def process_query(self, query: str, context: List[Chunk]) -> str:
    ...         # Implementation here
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Chunk:
    """
    Represents a chunk of text with associated metadata.

    Attributes:
        text: The text content of the chunk
        metadata: Additional metadata about the chunk
        score: Relevance score (0.0 to 1.0)
        source_id: Identifier for the source document
        chunk_id: Unique identifier for this chunk
    """
    text: str
    metadata: Dict[str, Any]
    score: float
    source_id: str
    chunk_id: str


@dataclass
class StrategyConfig:
    """
    Configuration parameters for RAG strategies.

    Attributes:
        chunk_size: Size of text chunks in tokens (default: 512)
        chunk_overlap: Overlap between chunks in tokens (default: 50)
        top_k: Number of results to retrieve (default: 5)
        strategy_name: Identifier for the strategy
        metadata: Additional strategy-specific parameters

    Raises:
        ValueError: If any parameter is invalid (e.g., negative values)
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size < 0:
            raise ValueError(f"chunk_size must be non-negative, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


@dataclass
class PreparedData:
    """
    Container for prepared data ready for retrieval.

    Attributes:
        chunks: List of text chunks
        embeddings: List of embedding vectors corresponding to chunks
        index_metadata: Metadata about the index (e.g., index type, version)
    """
    chunks: List[Chunk]
    embeddings: List[List[float]]
    index_metadata: Dict[str, Any]


@dataclass
class QueryResult:
    """
    Result from processing a query.

    Attributes:
        answer: The generated answer
        chunks_used: List of chunks used to generate the answer
        metadata: Additional metadata (e.g., execution time, confidence)
        strategy_info: Information about the strategy used
    """
    answer: str
    chunks_used: List[Chunk]
    metadata: Dict[str, Any]
    strategy_info: Dict[str, Any]


class IRAGStrategy(ABC):
    """
    Abstract base class for RAG strategies.

    This interface defines the contract that all RAG strategies must implement.
    It supports both synchronous and asynchronous operations, and provides
    methods for data preparation, retrieval, and query processing.

    All concrete implementations must provide implementations for:
    - initialize: Set up the strategy with configuration
    - prepare_data: Process and chunk documents for retrieval
    - retrieve: Synchronous retrieval of relevant chunks
    - aretrieve: Asynchronous retrieval of relevant chunks
    - process_query: Generate an answer from query and context

    Example:
        >>> config = StrategyConfig(chunk_size=1024, top_k=10)
        >>> strategy = MyConcreteStrategy()
        >>> strategy.initialize(config)
        >>> prepared = strategy.prepare_data(documents)
        >>> chunks = strategy.retrieve("What is RAG?", top_k=5)
        >>> answer = strategy.process_query("What is RAG?", chunks)
    """

    @abstractmethod
    def initialize(self, config: StrategyConfig) -> None:
        """
        Initialize the strategy with configuration.

        This method is called before any other operations and should set up
        all necessary resources and parameters.

        Args:
            config: Configuration parameters for the strategy

        Returns:
            None

        Example:
            >>> strategy = MyStrategy()
            >>> config = StrategyConfig(chunk_size=1024)
            >>> strategy.initialize(config)
        """
        ...

    @abstractmethod
    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """
        Prepare and chunk documents for retrieval.

        This method processes raw documents, chunks them according to the
        configuration, and prepares any necessary indices or embeddings.

        Args:
            documents: List of documents to prepare, each as a dictionary
                      with at least a 'text' field

        Returns:
            PreparedData: Container with chunks, embeddings, and metadata

        Example:
            >>> documents = [{"text": "Sample document", "id": "doc1"}]
            >>> prepared = strategy.prepare_data(documents)
            >>> print(len(prepared.chunks))
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """
        Retrieve relevant chunks based on query (synchronous).

        This method finds the most relevant chunks for a given query using
        the strategy's retrieval mechanism.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List[Chunk]: List of relevant chunks, sorted by relevance score

        Example:
            >>> chunks = strategy.retrieve("What is RAG?", top_k=5)
            >>> for chunk in chunks:
            ...     print(f"Score: {chunk.score}, Text: {chunk.text[:50]}")
        """
        ...

    @abstractmethod
    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """
        Retrieve relevant chunks based on query (asynchronous).

        This is the async version of retrieve(), useful for non-blocking
        operations and integration with async frameworks.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List[Chunk]: List of relevant chunks, sorted by relevance score

        Example:
            >>> chunks = await strategy.aretrieve("What is RAG?", top_k=5)
            >>> for chunk in chunks:
            ...     print(f"Score: {chunk.score}, Text: {chunk.text[:50]}")
        """
        ...

    @abstractmethod
    def process_query(self, query: str, context: List[Chunk]) -> str:
        """
        Process query with retrieved context to generate an answer.

        This method takes a query and relevant context chunks and generates
        a coherent answer using the strategy's processing logic.

        Args:
            query: The user's query
            context: List of relevant chunks to use as context

        Returns:
            str: The generated answer

        Example:
            >>> chunks = strategy.retrieve("What is RAG?", top_k=5)
            >>> answer = strategy.process_query("What is RAG?", chunks)
            >>> print(answer)
        """
        ...
