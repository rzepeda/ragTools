"""Standalone test to verify mock builders work correctly.

This test doesn't rely on pytest fixtures to avoid conftest.py dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.mocks import (
    create_mock_embedding_service,
    create_mock_database_service,
    create_mock_llm_service,
    create_mock_neo4j_service,
    create_mock_registry_with_services,
    create_mock_chunk,
)


def test_embedding_service():
    """Test embedding service mock."""
    service = create_mock_embedding_service(dimension=768)
    
    assert service.dimension == 768
    assert service.model_name == "mock-embedding-model"
    assert service.get_dimension() == 768
    print("✓ Embedding service mock works")


def test_database_service():
    """Test database service mock."""
    service = create_mock_database_service()
    
    assert service.get_context() == service
    assert hasattr(service, 'store_chunks')
    assert hasattr(service, 'search_chunks')
    print("✓ Database service mock works")


def test_llm_service():
    """Test LLM service mock."""
    service = create_mock_llm_service(generate_return_value="Test response")
    
    assert service.model_name == "mock-llm-model"
    assert hasattr(service, 'generate')
    print("✓ LLM service mock works")


def test_neo4j_service():
    """Test Neo4j service mock."""
    service = create_mock_neo4j_service()
    
    assert hasattr(service, 'execute_query')
    assert hasattr(service, 'close')
    print("✓ Neo4j service mock works")


def test_registry():
    """Test registry with services."""
    registry = create_mock_registry_with_services(
        include_embedding=True,
        include_database=True,
        include_llm=True,
        include_neo4j=True
    )
    
    # Verify all services are registered
    assert "embedding_local" in registry._instances
    assert "db_main" in registry._instances
    assert "llm_local" in registry._instances
    assert "db_neo4j" in registry._instances
    
    # Verify get method works
    embedding = registry.get("embedding_local")
    assert embedding is not None
    assert embedding.dimension == 384
    
    print("✓ Registry mock works")


def test_chunk():
    """Test chunk mock."""
    chunk = create_mock_chunk(
        id="test1",
        text="test content",
        score=0.95
    )
    
    assert chunk.id == "test1"
    assert chunk.text == "test content"
    assert chunk.score == 0.95
    print("✓ Chunk mock works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Centralized Mock System")
    print("="*60 + "\n")
    
    try:
        test_embedding_service()
        test_database_service()
        test_llm_service()
        test_neo4j_service()
        test_registry()
        test_chunk()
        
        print("\n" + "="*60)
        print("✅ All mock builders working correctly!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
