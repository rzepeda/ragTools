"""Unit tests for entity extractor."""

import pytest
from unittest.mock import Mock

from rag_factory.strategies.knowledge_graph.entity_extractor import EntityExtractor
from rag_factory.strategies.knowledge_graph.models import Entity, EntityType
from rag_factory.strategies.knowledge_graph.config import KnowledgeGraphConfig
from rag_factory.services.llm.base import LLMResponse


@pytest.fixture
def mock_llm():
    """Mock LLM service for testing."""
    llm = Mock()
    response = LLMResponse(
        content='''[
  {"name": "Python", "type": "concept", "description": "Programming language", "confidence": 0.95},
  {"name": "Machine Learning", "type": "concept", "description": "AI subset", "confidence": 0.90}
]''',
        model="test-model",
        provider="test-provider",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        latency=0.5,
        metadata={}
    )
    llm.complete.return_value = response
    return llm


@pytest.fixture
def config():
    """Default configuration for testing."""
    return KnowledgeGraphConfig(
        entity_types=[EntityType.CONCEPT, EntityType.PERSON],
        min_entity_confidence=0.5
    )


@pytest.fixture
def entity_extractor(mock_llm, config):
    """Entity extractor instance for testing."""
    return EntityExtractor(mock_llm, config)


def test_entity_extraction_basic(entity_extractor, mock_llm):
    """Test basic entity extraction."""
    text = "Python is great for Machine Learning applications."
    entities = entity_extractor.extract_entities(text, "chunk_1")
    
    assert len(entities) == 2
    assert entities[0].name == "Python"
    assert entities[0].type == EntityType.CONCEPT
    assert entities[1].name == "Machine Learning"
    assert mock_llm.complete.called


def test_entity_extraction_with_confidence(entity_extractor):
    """Test confidence scores."""
    entities = entity_extractor.extract_entities("Test text", "chunk_1")
    
    assert all(0.0 <= e.confidence <= 1.0 for e in entities)
    assert all(e.confidence >= 0.5 for e in entities)  # Min confidence filter


def test_entity_deduplication(entity_extractor):
    """Test entity deduplication."""
    entities = [
        Entity(
            id="e1",
            name="Python",
            type=EntityType.CONCEPT,
            confidence=0.9,
            source_chunks=["c1"]
        ),
        Entity(
            id="e2",
            name="python",  # Duplicate (case-insensitive)
            type=EntityType.CONCEPT,
            confidence=0.85,
            source_chunks=["c2"]
        ),
        Entity(
            id="e3",
            name="Java",
            type=EntityType.CONCEPT,
            confidence=0.8,
            source_chunks=["c1"]
        )
    ]
    
    unique = entity_extractor.deduplicate_entities(entities)
    
    assert len(unique) == 2  # Python and Java
    assert any(e.name.lower() == "python" for e in unique)
    assert any(e.name == "Java" for e in unique)
    
    # Check that source chunks were merged
    python_entity = next(e for e in unique if e.name.lower() == "python")
    assert len(python_entity.source_chunks) == 2


def test_batch_extraction(entity_extractor, mock_llm):
    """Test batch entity extraction."""
    texts = [
        {"text": "Text 1", "chunk_id": "c1"},
        {"text": "Text 2", "chunk_id": "c2"}
    ]
    
    results = entity_extractor.extract_entities_batch(texts)
    
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
    assert mock_llm.complete.call_count == 2


def test_invalid_json_response(entity_extractor, mock_llm):
    """Test handling of invalid JSON response."""
    response = LLMResponse(
        content="Invalid JSON",
        model="test-model",
        provider="test-provider",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        latency=0.5,
        metadata={}
    )
    mock_llm.complete.return_value = response
    
    entities = entity_extractor.extract_entities("Test", "chunk_1")
    
    assert len(entities) == 0  # Should return empty list


def test_confidence_filtering(mock_llm):
    """Test that low confidence entities are filtered."""
    config = KnowledgeGraphConfig(min_entity_confidence=0.8)
    extractor = EntityExtractor(mock_llm, config)
    
    response = LLMResponse(
        content='''[
  {"name": "High", "type": "concept", "description": "High confidence", "confidence": 0.9},
  {"name": "Low", "type": "concept", "description": "Low confidence", "confidence": 0.5}
]''',
        model="test-model",
        provider="test-provider",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        latency=0.5,
        metadata={}
    )
    mock_llm.complete.return_value = response
    
    entities = extractor.extract_entities("Test", "chunk_1")
    
    assert len(entities) == 1
    assert entities[0].name == "High"
