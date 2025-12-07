"""Unit tests for memory graph store."""

import pytest

from rag_factory.strategies.knowledge_graph.memory_graph_store import MemoryGraphStore
from rag_factory.strategies.knowledge_graph.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType
)


@pytest.fixture
def graph_store():
    """Graph store instance for testing."""
    return MemoryGraphStore()


def test_add_entity(graph_store):
    """Test adding entities to graph."""
    entity = Entity(
        id="e1",
        name="Python",
        type=EntityType.CONCEPT,
        description="Programming language",
        confidence=0.95
    )
    
    graph_store.add_entity(entity)
    
    retrieved = graph_store.get_entity("e1")
    assert retrieved is not None
    assert retrieved.name == "Python"


def test_add_relationship(graph_store):
    """Test adding relationships to graph."""
    # Add entities
    e1 = Entity(id="e1", name="Python", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="Machine Learning", type=EntityType.CONCEPT, confidence=1.0)
    
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)
    
    # Add relationship
    rel = Relationship(
        id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.RELATED_TO,
        strength=0.9,
        confidence=0.85
    )
    
    graph_store.add_relationship(rel)
    
    # Retrieve relationships
    relationships = graph_store.get_relationships("e1")
    assert len(relationships) >= 1
    assert relationships[0].target_entity_id == "e2"


def test_graph_traversal(graph_store):
    """Test graph traversal."""
    # Create a small graph: A -> B -> C
    entities = [
        Entity(id="A", name="Entity A", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="B", name="Entity B", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="C", name="Entity C", type=EntityType.CONCEPT, confidence=1.0)
    ]
    
    for entity in entities:
        graph_store.add_entity(entity)
    
    relationships = [
        Relationship(
            id="r1",
            source_entity_id="A",
            target_entity_id="B",
            type=RelationshipType.CONNECTED_TO,
            strength=1.0,
            confidence=1.0
        ),
        Relationship(
            id="r2",
            source_entity_id="B",
            target_entity_id="C",
            type=RelationshipType.CONNECTED_TO,
            strength=1.0,
            confidence=1.0
        )
    ]
    
    for rel in relationships:
        graph_store.add_relationship(rel)
    
    # Traverse from A with max_hops=2
    result = graph_store.traverse(["A"], max_hops=2)
    
    assert len(result.entities) == 3  # A, B, C
    assert len(result.relationships) == 2


def test_find_entities_by_name(graph_store):
    """Test entity search by name."""
    entities = [
        Entity(id="e1", name="Python Programming", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="e2", name="Java Programming", type=EntityType.CONCEPT, confidence=1.0),
        Entity(id="e3", name="Machine Learning", type=EntityType.CONCEPT, confidence=1.0)
    ]
    
    for entity in entities:
        graph_store.add_entity(entity)
    
    results = graph_store.find_entities_by_name("Programming")
    assert len(results) == 2
    assert all("Programming" in e.name for e in results)


def test_graph_stats(graph_store):
    """Test graph statistics."""
    # Add some entities and relationships
    e1 = Entity(id="e1", name="A", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="B", type=EntityType.CONCEPT, confidence=1.0)
    
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)
    
    rel = Relationship(
        id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.RELATED_TO,
        strength=1.0,
        confidence=1.0
    )
    graph_store.add_relationship(rel)
    
    stats = graph_store.get_stats()
    
    assert stats["num_entities"] == 2
    assert stats["num_relationships"] == 1
    assert "density" in stats
    assert "avg_degree" in stats


def test_delete_entity(graph_store):
    """Test entity deletion."""
    e1 = Entity(id="e1", name="A", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="B", type=EntityType.CONCEPT, confidence=1.0)
    
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)
    
    rel = Relationship(
        id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.RELATED_TO,
        strength=1.0,
        confidence=1.0
    )
    graph_store.add_relationship(rel)
    
    # Delete entity
    graph_store.delete_entity("e1")
    
    assert graph_store.get_entity("e1") is None
    assert len(graph_store.relationships) == 0  # Relationship should be deleted too


def test_clear_graph(graph_store):
    """Test clearing the graph."""
    e1 = Entity(id="e1", name="A", type=EntityType.CONCEPT, confidence=1.0)
    graph_store.add_entity(e1)
    
    graph_store.clear()
    
    assert len(graph_store.entities) == 0
    assert len(graph_store.relationships) == 0
    assert graph_store.graph.number_of_nodes() == 0


def test_relationship_type_filtering(graph_store):
    """Test filtering relationships by type."""
    e1 = Entity(id="e1", name="A", type=EntityType.CONCEPT, confidence=1.0)
    e2 = Entity(id="e2", name="B", type=EntityType.CONCEPT, confidence=1.0)
    
    graph_store.add_entity(e1)
    graph_store.add_entity(e2)
    
    rel1 = Relationship(
        id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.RELATED_TO,
        strength=1.0,
        confidence=1.0
    )
    rel2 = Relationship(
        id="r2",
        source_entity_id="e1",
        target_entity_id="e2",
        type=RelationshipType.CAUSES,
        strength=1.0,
        confidence=1.0
    )
    
    graph_store.add_relationship(rel1)
    graph_store.add_relationship(rel2)
    
    # Filter by type
    related_to_rels = graph_store.get_relationships("e1", relationship_type="related_to")
    assert len(related_to_rels) == 1
    assert related_to_rels[0].type == RelationshipType.RELATED_TO
