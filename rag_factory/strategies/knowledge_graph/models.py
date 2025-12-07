"""
Data models for Knowledge Graph RAG Strategy.

This module defines the core data structures for entities, relationships,
and graph traversal results used in the knowledge graph strategy.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """Entity types supported by the knowledge graph."""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Relationship types supported by the knowledge graph."""
    IS_PART_OF = "is_part_of"
    IS_A = "is_a"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    BEFORE = "before"
    AFTER = "after"
    CONNECTED_TO = "connected_to"
    BELONGS_TO = "belongs_to"
    CUSTOM = "custom"


class Entity(BaseModel):
    """
    Entity node in knowledge graph.
    
    Attributes:
        id: Unique entity identifier
        name: Entity name
        type: Entity type (person, place, concept, etc.)
        description: Optional description of the entity
        properties: Additional entity properties
        confidence: Confidence score for entity extraction (0.0-1.0)
        source_chunks: List of chunk IDs where this entity appears
    """
    id: str = Field(..., description="Unique entity ID")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunks: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """
    Relationship edge in knowledge graph.
    
    Attributes:
        id: Unique relationship identifier
        source_entity_id: ID of source entity
        target_entity_id: ID of target entity
        type: Relationship type
        description: Optional description of the relationship
        strength: Relationship strength (0.0-1.0)
        confidence: Confidence score for relationship extraction (0.0-1.0)
        properties: Additional relationship properties
        source_chunks: List of chunk IDs where this relationship appears
    """
    id: str = Field(..., description="Unique relationship ID")
    source_entity_id: str
    target_entity_id: str
    type: RelationshipType
    description: Optional[str] = None
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_chunks: List[str] = Field(default_factory=list)


class GraphTraversalResult(BaseModel):
    """
    Result from graph traversal operation.
    
    Attributes:
        entities: List of entities found during traversal
        relationships: List of relationships found during traversal
        paths: List of entity ID paths (each path is a list of entity IDs)
        scores: Dictionary mapping entity IDs to relevance scores
    """
    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]]  # List of entity ID paths
    scores: Dict[str, float]  # Entity ID -> relevance score


class HybridSearchResult(BaseModel):
    """
    Result from hybrid vector + graph search.
    
    Attributes:
        chunk_id: Chunk identifier
        text: Chunk text content
        vector_score: Score from vector similarity search
        graph_score: Score from graph connectivity
        combined_score: Weighted combination of vector and graph scores
        related_entities: Entities found in this chunk
        relationship_paths: Paths through the graph involving these entities
        metadata: Additional metadata
    """
    chunk_id: str
    text: str
    vector_score: float
    graph_score: float
    combined_score: float
    related_entities: List[Entity]
    relationship_paths: List[List[str]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
