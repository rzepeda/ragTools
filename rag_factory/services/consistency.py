"""Consistency checking for RAG strategies.

This module provides validation logic to ensure that strategies declare
consistent capabilities and service dependencies. The checker warns about
potential misconfigurations without blocking strategy creation.
"""

import logging
from typing import List, Set, TYPE_CHECKING

from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency

if TYPE_CHECKING:
    from typing import Protocol
    
    class IndexingStrategy(Protocol):
        """Protocol for indexing strategies."""
        def produces(self) -> Set[IndexCapability]: ...
        def requires_services(self) -> Set[ServiceDependency]: ...
        @property
        def __class__(self): ...
    
    class RetrievalStrategy(Protocol):
        """Protocol for retrieval strategies."""
        def requires(self) -> Set[IndexCapability]: ...
        def requires_services(self) -> Set[ServiceDependency]: ...
        @property
        def __class__(self): ...


logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Checks consistency between capabilities and services.
    
    This class validates that strategies declare consistent relationships
    between the capabilities they produce/require and the services they
    depend on. It generates warnings for potential misconfigurations but
    does not block strategy creation, allowing flexibility for edge cases.
    
    Example:
        >>> checker = ConsistencyChecker()
        >>> warnings = checker.check_indexing_strategy(my_strategy)
        >>> for warning in warnings:
        ...     logger.warning(warning)
    """
    
    def check_indexing_strategy(self, strategy: "IndexingStrategy") -> List[str]:
        """Check consistency for an indexing strategy.
        
        Validates that the capabilities produced by the strategy are consistent
        with its declared service dependencies. Common checks include:
        - VECTORS capability should require EMBEDDING service
        - GRAPH capability should require GRAPH service
        - DATABASE capability should require DATABASE service
        - IN_MEMORY capability should NOT require DATABASE service (usually)
        
        Args:
            strategy: The indexing strategy to check. Must implement:
                     - produces() -> Set[IndexCapability]
                     - requires_services() -> Set[ServiceDependency]
        
        Returns:
            List of warning messages for inconsistencies found.
            Empty list if strategy is consistent.
        
        Example:
            >>> class MyIndexer:
            ...     def produces(self):
            ...         return {IndexCapability.VECTORS}
            ...     def requires_services(self):
            ...         return set()  # Missing EMBEDDING!
            >>> checker = ConsistencyChecker()
            >>> warnings = checker.check_indexing_strategy(MyIndexer())
            >>> print(warnings[0])
            ⚠️ MyIndexer: Produces VECTORS but doesn't require EMBEDDING service.
        """
        warnings = []
        produces = strategy.produces()
        requires_services = strategy.requires_services()
        name = strategy.__class__.__name__
        
        # Check VECTORS capability requires EMBEDDING service
        if IndexCapability.VECTORS in produces and ServiceDependency.EMBEDDING not in requires_services:
            warnings.append(
                f"⚠️ {name}: Produces VECTORS but doesn't require EMBEDDING service. "
                f"Vector embeddings typically need an embedding service to generate."
            )
        
        # Check GRAPH capability requires GRAPH service
        if IndexCapability.GRAPH in produces and ServiceDependency.GRAPH not in requires_services:
            warnings.append(
                f"⚠️ {name}: Produces GRAPH but doesn't require GRAPH service. "
                f"Knowledge graphs typically need a graph database service."
            )
        
        # Check DATABASE capability requires DATABASE service
        if IndexCapability.DATABASE in produces and ServiceDependency.DATABASE not in requires_services:
            warnings.append(
                f"⚠️ {name}: Produces DATABASE capability but doesn't require DATABASE service. "
                f"Database persistence typically needs a database service."
            )
        
        # Warn if IN_MEMORY capability requires DATABASE service (unusual)
        if IndexCapability.IN_MEMORY in produces and ServiceDependency.DATABASE in requires_services:
            warnings.append(
                f"⚠️ {name}: Produces IN_MEMORY capability but requires DATABASE service. "
                f"In-memory storage usually doesn't need database persistence."
            )
        
        return warnings
    
    def check_retrieval_strategy(self, strategy: "RetrievalStrategy") -> List[str]:
        """Check consistency for a retrieval strategy.
        
        Validates that the capabilities required by the strategy are consistent
        with its declared service dependencies. Common checks include:
        - VECTORS requirement should require DATABASE service (for search)
        - GRAPH requirement should require GRAPH service
        - KEYWORDS requirement should require DATABASE service
        
        Args:
            strategy: The retrieval strategy to check. Must implement:
                     - requires() -> Set[IndexCapability]
                     - requires_services() -> Set[ServiceDependency]
        
        Returns:
            List of warning messages for inconsistencies found.
            Empty list if strategy is consistent.
        
        Example:
            >>> class MyRetriever:
            ...     def requires(self):
            ...         return {IndexCapability.VECTORS}
            ...     def requires_services(self):
            ...         return {ServiceDependency.LLM}  # Missing DATABASE!
            >>> checker = ConsistencyChecker()
            >>> warnings = checker.check_retrieval_strategy(MyRetriever())
            >>> print(warnings[0])
            ⚠️ MyRetriever: Requires VECTORS capability but doesn't require DATABASE service.
        """
        warnings = []
        requires_caps = strategy.requires()
        requires_services = strategy.requires_services()
        name = strategy.__class__.__name__
        
        # Check VECTORS requirement requires DATABASE service
        if IndexCapability.VECTORS in requires_caps and ServiceDependency.DATABASE not in requires_services:
            warnings.append(
                f"⚠️ {name}: Requires VECTORS capability but doesn't require DATABASE service. "
                f"Vector search typically needs a database service to query embeddings."
            )
        
        # Check GRAPH requirement requires GRAPH service
        if IndexCapability.GRAPH in requires_caps and ServiceDependency.GRAPH not in requires_services:
            warnings.append(
                f"⚠️ {name}: Requires GRAPH capability but doesn't require GRAPH service. "
                f"Graph traversal typically needs a graph database service."
            )
        
        # Check KEYWORDS requirement requires DATABASE service
        if IndexCapability.KEYWORDS in requires_caps and ServiceDependency.DATABASE not in requires_services:
            warnings.append(
                f"⚠️ {name}: Requires KEYWORDS capability but doesn't require DATABASE service. "
                f"Keyword search typically needs a database service to query the index."
            )
        
        return warnings
