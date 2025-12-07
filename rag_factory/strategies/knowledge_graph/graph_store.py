"""
Abstract interface for graph storage backends.

This module defines the abstract base class that all graph storage
implementations must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .models import Entity, Relationship, GraphTraversalResult


class GraphStore(ABC):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """
        Add entity node to graph.
        
        Args:
            entity: Entity to add
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add relationship edge to graph.
        
        Args:
            relationship: Relationship to add
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity identifier
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    def traverse(
        self,
        start_entity_ids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> GraphTraversalResult:
        """
        Traverse graph from starting entities.
        
        Args:
            start_entity_ids: Starting entity IDs
            max_hops: Maximum traversal depth
            relationship_types: Optional filter by relationship types
            
        Returns:
            Graph traversal result
        """
        pass

    @abstractmethod
    def find_entities_by_name(self, name_pattern: str) -> List[Entity]:
        """
        Find entities matching name pattern.
        
        Args:
            name_pattern: Name pattern to search for
            
        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> None:
        """
        Delete entity and its relationships.
        
        Args:
            entity_id: Entity identifier
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from graph."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        pass
