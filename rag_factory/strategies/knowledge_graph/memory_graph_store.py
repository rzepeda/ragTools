"""
In-memory graph store using NetworkX.

This module implements the graph storage interface using NetworkX
for in-memory graph operations.
"""

from typing import List, Dict, Any, Optional
import networkx as nx
import logging

from .graph_store import GraphStore
from .models import Entity, Relationship, GraphTraversalResult

logger = logging.getLogger(__name__)


class MemoryGraphStore(GraphStore):
    """In-memory graph store using NetworkX."""

    def __init__(self):
        """Initialize memory graph store."""
        self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to support multiple relationships
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add entity node to graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type.value,
            description=entity.description,
            confidence=entity.confidence,
            properties=entity.properties
        )
        logger.debug(f"Added entity: {entity.name} ({entity.id})")

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship edge to graph."""
        self.relationships[relationship.id] = relationship
        self.graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            id=relationship.id,
            type=relationship.type.value,
            description=relationship.description,
            strength=relationship.strength,
            confidence=relationship.confidence,
            properties=relationship.properties
        )
        logger.debug(
            f"Added relationship: {relationship.source_entity_id} "
            f"-[{relationship.type.value}]-> {relationship.target_entity_id}"
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        relationships = []
        
        if entity_id not in self.graph:
            return relationships
        
        # Outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            rel_id = data.get("id")
            rel = self.relationships.get(rel_id)
            if rel and (not relationship_type or rel.type.value == relationship_type):
                relationships.append(rel)
        
        # Incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            rel_id = data.get("id")
            rel = self.relationships.get(rel_id)
            if rel and (not relationship_type or rel.type.value == relationship_type):
                relationships.append(rel)
        
        return relationships

    def traverse(
        self,
        start_entity_ids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> GraphTraversalResult:
        """Traverse graph from starting entities using BFS."""
        visited_entities = set()
        visited_relationships = set()
        paths = []
        
        for start_id in start_entity_ids:
            if start_id not in self.graph:
                logger.warning(f"Entity {start_id} not in graph")
                continue
            
            # BFS traversal with hop limit
            visited = {start_id: 0}  # entity_id -> hop count
            queue = [(start_id, 0, [start_id])]
            
            while queue:
                current_id, hops, path = queue.pop(0)
                
                visited_entities.add(current_id)
                paths.append(path)
                
                # Stop expanding if we've reached max hops
                if hops >= max_hops:
                    continue
                
                # Get neighbors (for MultiDiGraph, we need to handle multiple edges)
                for neighbor in self.graph.neighbors(current_id):
                    # Get all edges between current and neighbor
                    for key, edge_data in self.graph[current_id][neighbor].items():
                        edge_type = edge_data.get("type")
                        
                        if relationship_types and edge_type not in relationship_types:
                            continue
                        
                        if neighbor not in visited or visited[neighbor] > hops + 1:
                            visited[neighbor] = hops + 1
                            queue.append((neighbor, hops + 1, path + [neighbor]))
                        
                        # Track relationship
                        rel_id = edge_data.get("id")
                        if rel_id:
                            visited_relationships.add(rel_id)
        
        # Collect entities and relationships
        entities = [self.entities[eid] for eid in visited_entities if eid in self.entities]
        relationships = [
            self.relationships[rid]
            for rid in visited_relationships
            if rid in self.relationships
        ]
        
        # Calculate scores (simple degree centrality)
        scores = {}
        for entity_id in visited_entities:
            if entity_id in self.graph:
                degree = self.graph.degree(entity_id)
                scores[entity_id] = float(degree) / (len(self.graph.nodes) or 1)
        
        return GraphTraversalResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            scores=scores
        )

    def find_entities_by_name(self, name_pattern: str) -> List[Entity]:
        """Find entities matching name pattern."""
        pattern_lower = name_pattern.lower()
        matching = []
        
        for entity in self.entities.values():
            if pattern_lower in entity.name.lower():
                matching.append(entity)
        
        return matching

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity and its relationships."""
        if entity_id in self.graph:
            # Remove associated relationships
            for _, _, data in self.graph.edges(entity_id, data=True):
                rel_id = data.get("id")
                if rel_id and rel_id in self.relationships:
                    del self.relationships[rel_id]
            
            self.graph.remove_node(entity_id)
            if entity_id in self.entities:
                del self.entities[entity_id]
            logger.debug(f"Deleted entity: {entity_id}")

    def clear(self) -> None:
        """Clear all data from graph."""
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        logger.info("Graph cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": nx.density(self.graph) if num_nodes > 0 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / (num_nodes or 1)
        }
