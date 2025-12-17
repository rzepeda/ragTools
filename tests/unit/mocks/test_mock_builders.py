"""Unit tests for centralized mock builders.

These tests verify that the mock builders create properly configured
mock objects with the expected behavior.
"""

import pytest
from unittest.mock import AsyncMock

from tests.mocks import (
    create_mock_embedding_service,
    create_mock_database_service,
    create_mock_llm_service,
    create_mock_neo4j_service,
    create_mock_registry_with_services,
    create_mock_chunk,
    create_mock_document,
)


class TestServiceMocks:
    """Test service mock builders."""
    
    def test_create_mock_embedding_service(self):
        """Test embedding service mock creation."""
        service = create_mock_embedding_service(dimension=768)
        
        assert service.dimension == 768
        assert service.model_name == "mock-embedding-model"
        assert isinstance(service.embed, AsyncMock)
        assert isinstance(service.embed_batch, AsyncMock)
        assert service.get_dimension() == 768
    
    def test_create_mock_database_service(self):
        """Test database service mock creation."""
        service = create_mock_database_service()
        
        assert isinstance(service.store_chunks, AsyncMock)
        assert isinstance(service.search_chunks, AsyncMock)
        assert isinstance(service.get_chunks_for_documents, AsyncMock)
        assert service.get_context() == service
    
    def test_create_mock_llm_service(self):
        """Test LLM service mock creation."""
        service = create_mock_llm_service(
            generate_return_value="Custom response"
        )
        
        assert isinstance(service.generate, AsyncMock)
        assert isinstance(service.agenerate, AsyncMock)
        assert service.model_name == "mock-llm-model"
    
    def test_create_mock_neo4j_service(self):
        """Test Neo4j service mock creation."""
        service = create_mock_neo4j_service()
        
        assert isinstance(service.execute_query, AsyncMock)
        assert isinstance(service.close, AsyncMock)


class TestRegistryMocks:
    """Test registry mock builders."""
    
    def test_create_mock_registry_with_services(self):
        """Test registry creation with services."""
        registry = create_mock_registry_with_services(
            include_embedding=True,
            include_database=True,
            include_llm=True
        )
        
        # Verify services are registered
        assert "embedding_local" in registry._instances
        assert "db_main" in registry._instances
        assert "llm_local" in registry._instances
        
        # Verify get method works
        embedding_service = registry.get("embedding_local")
        assert embedding_service is not None
        assert embedding_service.dimension == 384
    
    def test_create_mock_registry_with_graph_services(self):
        """Test registry creation with graph services."""
        registry = create_mock_registry_with_services(
            include_neo4j=True
        )
        
        assert "db_neo4j" in registry._instances
        neo4j_service = registry.get("db_neo4j")
        assert neo4j_service is not None


class TestDataMocks:
    """Test data object mock builders."""
    
    def test_create_mock_chunk(self):
        """Test chunk mock creation."""
        chunk = create_mock_chunk(
            id="test1",
            text="test content",
            score=0.95
        )
        
        assert chunk.id == "test1"
        assert chunk.text == "test content"
        assert chunk.score == 0.95
        assert len(chunk.embedding) == 384
    
    def test_create_mock_document(self):
        """Test document mock creation."""
        doc = create_mock_document(
            id="doc1",
            filename="test.pdf",
            content="document content"
        )
        
        assert doc.id == "doc1"
        assert doc.filename == "test.pdf"
        assert doc.content == "document content"


class TestFixtures:
    """Test pytest fixtures."""
    
    def test_mock_embedding_service_fixture(self, mock_embedding_service):
        """Test mock_embedding_service fixture."""
        assert mock_embedding_service.dimension == 384
        assert isinstance(mock_embedding_service.embed, AsyncMock)
    
    def test_mock_database_service_fixture(self, mock_database_service):
        """Test mock_database_service fixture."""
        assert isinstance(mock_database_service.store_chunks, AsyncMock)
    
    def test_mock_llm_service_fixture(self, mock_llm_service):
        """Test mock_llm_service fixture."""
        assert isinstance(mock_llm_service.generate, AsyncMock)
    
    def test_mock_registry_with_services_fixture(self, mock_registry_with_services):
        """Test mock_registry_with_services fixture."""
        assert "embedding_local" in mock_registry_with_services._instances
        assert "db_main" in mock_registry_with_services._instances
        
        embedding = mock_registry_with_services.get("embedding_local")
        assert embedding is not None
