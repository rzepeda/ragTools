"""Retrieval strategy interface and context.

This module defines the interface for document retrieval strategies and
the shared context used during retrieval operations. It enables separation
of retrieval from indexing pipelines and ensures compatibility validation.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, TYPE_CHECKING

from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies

if TYPE_CHECKING:
    from rag_factory.services.interfaces import IDatabaseService
    from rag_factory.strategies.base import Chunk


class RetrievalContext:
    """Shared context for retrieval operations.

    This class holds shared resources and configuration that are used
    across retrieval operations. It provides a centralized place for:
    - Database service access
    - Configuration parameters
    - Performance metrics tracking

    Attributes:
        database: Database service for retrieving indexed data
        config: Configuration dictionary for retrieval parameters
        metrics: Dictionary for tracking performance metrics

    Example:
        >>> from unittest.mock import Mock
        >>> db_service = Mock()
        >>> context = RetrievalContext(
        ...     database_service=db_service,
        ...     config={"top_k": 10}
        ... )
        >>> context.metrics["chunks_retrieved"] = 10
        >>> context.config["top_k"]
        10
    """

    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: Dict[str, Any] = None
    ):
        """Initialize retrieval context.

        Args:
            database_service: Database service for retrieving indexed data
            config: Optional configuration dictionary
        """
        self.database = database_service
        self.config = config or {}
        self.metrics: Dict[str, Any] = {}


class IRetrievalStrategy(ABC):
    """Interface for document retrieval strategies.

    This abstract base class defines the contract that all retrieval
    strategies must implement. It enforces:
    - Dependency declaration and validation
    - Capability requirement declaration (what the strategy needs)
    - Async document retrieval

    Retrieval strategies are responsible for finding and ranking relevant
    chunks from the indexed data based on a user query.

    Example:
        >>> class MyRetrievalStrategy(IRetrievalStrategy):
        ...     def requires(self) -> Set[IndexCapability]:
        ...         return {IndexCapability.VECTORS, IndexCapability.DATABASE}
        ...
        ...     def requires_services(self) -> Set[ServiceDependency]:
        ...         return {ServiceDependency.DATABASE}
        ...
        ...     async def retrieve(self, query, context, top_k=10):
        ...         # Implementation here
        ...         return []
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """Initialize retrieval strategy.

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
            ...     database_service=Mock()
            ... )
            >>> config = {"top_k": 10}
            >>> # This would work for a strategy requiring DATABASE
        """
        self.config = config
        self.deps = dependencies

        # Validate dependencies
        from rag_factory.services.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())

    @abstractmethod
    def requires(self) -> Set[IndexCapability]:
        """Declare what index capabilities this strategy requires.

        This method must return the set of capabilities that must be
        available from the indexing phase for this retrieval strategy
        to function properly.

        Returns:
            Set of IndexCapability enums representing what this strategy requires

        Example:
            >>> def requires(self) -> Set[IndexCapability]:
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
            ...         ServiceDependency.DATABASE,
            ...         ServiceDependency.RERANKER
            ...     }
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> List['Chunk']:
        """Retrieve relevant chunks for query.

        This method performs the actual retrieval work. It should:
        1. Process the query
        2. Search the indexed data using the context's database service
        3. Rank and filter results
        4. Return the most relevant chunks

        Args:
            query: User query string
            context: Shared retrieval context with database service and config
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores, ordered by relevance

        Example:
            >>> async def retrieve(self, query, context, top_k=10):
            ...     # Search database
            ...     results = await context.database.search_vectors(
            ...         query_embedding=self._embed_query(query),
            ...         top_k=top_k
            ...     )
            ...
            ...     # Update metrics
            ...     context.metrics["chunks_retrieved"] = len(results)
            ...
            ...     return results
        """
        pass


class RerankingRetrieval(IRetrievalStrategy):
    """Example retrieval strategy using vector search and reranking.

    This is a reference implementation that demonstrates how to:
    - Declare capability and service requirements
    - Validate dependencies
    - Use injected services
    - Perform multi-step retrieval (search + rerank)
    - Use the retrieval context

    This strategy:
    - Performs initial vector search to get candidate chunks
    - Reranks candidates using a reranking service
    - Returns top-k most relevant chunks

    Requires (Capabilities):
        - VECTORS: Vector embeddings for semantic search
        - CHUNKS: Document chunks to retrieve
        - DATABASE: Data persisted in database

    Requires (Services):
        - DATABASE: Service for searching vectors
        - RERANKER: Service for reranking results

    Example:
        >>> from unittest.mock import Mock, AsyncMock
        >>> database_service = Mock()
        >>> database_service.search_vectors = AsyncMock(return_value=[])
        >>> reranker_service = Mock()
        >>> reranker_service.rerank = AsyncMock(return_value=[])
        >>>
        >>> deps = StrategyDependencies(
        ...     database_service=database_service,
        ...     reranker_service=reranker_service
        ... )
        >>> config = {"initial_k": 50, "final_k": 10}
        >>> strategy = RerankingRetrieval(config, deps)
        >>>
        >>> context = RetrievalContext(database_service, config)
        >>> # results = await strategy.retrieve("query", context, top_k=10)
    """

    def requires(self) -> Set[IndexCapability]:
        """Declare required capabilities.

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
            Set containing DATABASE and RERANKER service dependencies
        """
        return {
            ServiceDependency.DATABASE,
            ServiceDependency.RERANKER
        }

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> List['Chunk']:
        """Retrieve and rerank relevant chunks.

        Performs two-stage retrieval:
        1. Vector search to get initial candidates
        2. Reranking to refine results

        Args:
            query: User query string
            context: Retrieval context with database service
            top_k: Number of final results to return

        Returns:
            List of reranked chunks, ordered by relevance
        """
        # Get configuration
        initial_k = self.config.get("initial_k", top_k * 5)

        # Stage 1: Vector search for initial candidates
        candidates = await context.database.search_vectors(
            query=query,
            top_k=initial_k
        )

        # Update metrics
        context.metrics["initial_candidates"] = len(candidates)

        # Stage 2: Rerank candidates
        if candidates:
            reranked = await self.deps.reranker_service.rerank(
                query=query,
                chunks=candidates,
                top_k=top_k
            )
        else:
            reranked = []

        # Update metrics
        context.metrics["final_results"] = len(reranked)

        return reranked
