"""Dependency injection container for RAG strategies.

This module provides a container for service dependencies and validation
logic to ensure strategies have all required services before execution.
"""

from dataclasses import dataclass
from typing import Optional, Set, Tuple, List
from enum import Enum, auto

from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IDatabaseService,
    IRerankingService,
)


class ServiceDependency(Enum):
    """Services that strategies may depend on.

    This enum defines all available service types that can be injected
    into RAG strategies. Use these values to declare required dependencies.

    Example:
        >>> required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        >>> deps = StrategyDependencies(llm_service=my_llm, embedding_service=my_embedder)
        >>> is_valid, missing = deps.validate_for_strategy(required)
    """

    LLM = auto()
    EMBEDDING = auto()
    GRAPH = auto()
    DATABASE = auto()
    RERANKER = auto()


@dataclass
class StrategyDependencies:
    """Container for injected services.

    This dataclass holds instances of service interfaces that can be
    injected into RAG strategies. All fields are optional to allow
    partial dependency injection.

    Attributes:
        llm_service: Large Language Model service for text generation
        embedding_service: Service for generating text embeddings
        graph_service: Graph database service for knowledge graphs
        database_service: Database service for storing and retrieving chunks
        reranker_service: Service for reranking documents by relevance

    Example:
        >>> deps = StrategyDependencies(
        ...     llm_service=my_llm,
        ...     embedding_service=my_embedder
        ... )
        >>> required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        >>> is_valid, missing = deps.validate_for_strategy(required)
        >>> if not is_valid:
        ...     print(deps.get_missing_services_message(required))
    """

    llm_service: Optional[ILLMService] = None
    embedding_service: Optional[IEmbeddingService] = None
    graph_service: Optional[IGraphService] = None
    database_service: Optional[IDatabaseService] = None
    reranker_service: Optional[IRerankingService] = None

    def validate_for_strategy(
        self,
        required_services: Set[ServiceDependency]
    ) -> Tuple[bool, List[ServiceDependency]]:
        """Validate that all required services are present.

        Checks if all services specified in required_services are available
        in this container. Returns validation status and list of missing services.

        Args:
            required_services: Set of ServiceDependency enums representing
                             the services required by a strategy

        Returns:
            Tuple of (is_valid, missing_services) where:
            - is_valid: True if all required services are present, False otherwise
            - missing_services: List of ServiceDependency enums for missing services

        Example:
            >>> deps = StrategyDependencies(llm_service=my_llm)
            >>> required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
            >>> is_valid, missing = deps.validate_for_strategy(required)
            >>> print(is_valid)  # False
            >>> print(missing)   # [ServiceDependency.EMBEDDING]
        """
        missing = []

        if ServiceDependency.LLM in required_services and not self.llm_service:
            missing.append(ServiceDependency.LLM)
        if ServiceDependency.EMBEDDING in required_services and not self.embedding_service:
            missing.append(ServiceDependency.EMBEDDING)
        if ServiceDependency.GRAPH in required_services and not self.graph_service:
            missing.append(ServiceDependency.GRAPH)
        if ServiceDependency.DATABASE in required_services and not self.database_service:
            missing.append(ServiceDependency.DATABASE)
        if ServiceDependency.RERANKER in required_services and not self.reranker_service:
            missing.append(ServiceDependency.RERANKER)

        return (len(missing) == 0, missing)

    def get_missing_services_message(
        self,
        required_services: Set[ServiceDependency]
    ) -> str:
        """Get user-friendly error message for missing services.

        Generates a formatted error message listing all missing required services.
        Returns empty string if all required services are present.

        Args:
            required_services: Set of ServiceDependency enums representing
                             the services required by a strategy

        Returns:
            Error message string listing missing services, or empty string
            if validation passes

        Example:
            >>> deps = StrategyDependencies(llm_service=my_llm)
            >>> required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
            >>> message = deps.get_missing_services_message(required)
            >>> print(message)  # "Missing required services: EMBEDDING"
        """
        is_valid, missing = self.validate_for_strategy(required_services)

        if is_valid:
            return ""

        service_names = [s.name for s in missing]
        return f"Missing required services: {', '.join(service_names)}"


def validate_dependencies(
    dependencies: StrategyDependencies,
    required_services: Set[ServiceDependency]
) -> None:
    """Validate that all required services are present.
    
    This function checks if all services specified in required_services
    are available in the dependencies container. If any services are
    missing, it raises a ValueError with a descriptive error message.
    
    Args:
        dependencies: Container with injected services
        required_services: Set of ServiceDependency enums representing
                         the services required by a strategy
    
    Raises:
        ValueError: If any required services are missing, with a message
                   listing all missing services
    
    Example:
        >>> deps = StrategyDependencies(llm_service=my_llm)
        >>> required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        >>> validate_dependencies(deps, required)  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: Missing required services: EMBEDDING
        
        >>> deps = StrategyDependencies(
        ...     llm_service=my_llm,
        ...     embedding_service=my_embedder
        ... )
        >>> validate_dependencies(deps, required)  # Passes
    """
    is_valid, missing = dependencies.validate_for_strategy(required_services)
    
    if not is_valid:
        service_names = [s.name for s in missing]
        raise ValueError(f"Missing required services: {', '.join(service_names)}")

