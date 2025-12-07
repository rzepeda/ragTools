"""
Configuration for Knowledge Graph RAG Strategy.

This module defines configuration classes for the knowledge graph strategy,
including entity/relationship types, graph backend settings, and hybrid search parameters.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .models import EntityType, RelationshipType


class KnowledgeGraphConfig(BaseModel):
    """
    Configuration for Knowledge Graph RAG Strategy.
    
    Attributes:
        entity_types: List of entity types to extract
        relationship_types: List of relationship types to extract
        graph_backend: Graph storage backend ("memory" or "neo4j")
        vector_weight: Weight for vector similarity score (0.0-1.0)
        graph_weight: Weight for graph connectivity score (0.0-1.0)
        max_graph_hops: Maximum hops for graph traversal
        min_entity_confidence: Minimum confidence for entity extraction
        min_relationship_confidence: Minimum confidence for relationship extraction
        min_relationship_strength: Minimum strength for relationship extraction
        batch_size: Batch size for entity extraction
        enable_entity_deduplication: Whether to deduplicate entities
        neo4j_config: Neo4j configuration (if using Neo4j backend)
    """
    # Entity and relationship configuration
    entity_types: List[EntityType] = Field(
        default_factory=lambda: [
            EntityType.PERSON,
            EntityType.PLACE,
            EntityType.ORGANIZATION,
            EntityType.CONCEPT,
            EntityType.EVENT
        ],
        description="Entity types to extract"
    )
    
    relationship_types: List[RelationshipType] = Field(
        default_factory=lambda: [
            RelationshipType.IS_PART_OF,
            RelationshipType.IS_A,
            RelationshipType.RELATED_TO,
            RelationshipType.CAUSES,
            RelationshipType.CONNECTED_TO
        ],
        description="Relationship types to extract"
    )
    
    # Graph backend configuration
    graph_backend: str = Field(
        default="memory",
        description="Graph storage backend (memory or neo4j)"
    )
    
    # Hybrid search configuration
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity score"
    )
    
    graph_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for graph connectivity score"
    )
    
    max_graph_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum hops for graph traversal"
    )
    
    # Extraction thresholds
    min_entity_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for entity extraction"
    )
    
    min_relationship_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for relationship extraction"
    )
    
    min_relationship_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum strength for relationship extraction"
    )
    
    # Processing configuration
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for entity extraction"
    )
    
    enable_entity_deduplication: bool = Field(
        default=True,
        description="Whether to deduplicate entities"
    )
    
    # Neo4j configuration (optional)
    neo4j_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Neo4j configuration (uri, user, password)"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        # Ensure weights sum to 1.0 (or close to it)
        total_weight = self.vector_weight + self.graph_weight
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.vector_weight = self.vector_weight / total_weight
            self.graph_weight = self.graph_weight / total_weight
