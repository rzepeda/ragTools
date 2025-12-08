"""Unit tests for dependency injection container.

Tests for ServiceDependency enum and StrategyDependencies dataclass
including validation logic and error messaging.
"""

import pytest
from typing import List
from unittest.mock import Mock

from rag_factory.services.dependencies import (
    ServiceDependency,
    StrategyDependencies,
)
from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IDatabaseService,
    IRerankingService,
)


class TestServiceDependency:
    """Tests for ServiceDependency enum."""
    
    def test_enum_has_all_service_types(self):
        """Test that enum defines all 5 service types."""
        assert hasattr(ServiceDependency, 'LLM')
        assert hasattr(ServiceDependency, 'EMBEDDING')
        assert hasattr(ServiceDependency, 'GRAPH')
        assert hasattr(ServiceDependency, 'DATABASE')
        assert hasattr(ServiceDependency, 'RERANKER')
    
    def test_enum_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in ServiceDependency]
        assert len(values) == len(set(values))
    
    def test_enum_count(self):
        """Test that enum has exactly 5 members."""
        assert len(ServiceDependency) == 5


class TestStrategyDependencies:
    """Tests for StrategyDependencies container."""
    
    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create mock LLM service."""
        return Mock(spec=ILLMService)
    
    @pytest.fixture
    def mock_embedding(self) -> Mock:
        """Create mock embedding service."""
        return Mock(spec=IEmbeddingService)
    
    @pytest.fixture
    def mock_graph(self) -> Mock:
        """Create mock graph service."""
        return Mock(spec=IGraphService)
    
    @pytest.fixture
    def mock_database(self) -> Mock:
        """Create mock database service."""
        return Mock(spec=IDatabaseService)
    
    @pytest.fixture
    def mock_reranker(self) -> Mock:
        """Create mock reranking service."""
        return Mock(spec=IRerankingService)
    
    def test_instantiation_with_no_services(self):
        """Test creating container with no services."""
        deps = StrategyDependencies()
        
        assert deps.llm_service is None
        assert deps.embedding_service is None
        assert deps.graph_service is None
        assert deps.database_service is None
        assert deps.reranker_service is None
    
    def test_instantiation_with_all_services(
        self,
        mock_llm,
        mock_embedding,
        mock_graph,
        mock_database,
        mock_reranker
    ):
        """Test creating container with all services."""
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding,
            graph_service=mock_graph,
            database_service=mock_database,
            reranker_service=mock_reranker
        )
        
        assert deps.llm_service is mock_llm
        assert deps.embedding_service is mock_embedding
        assert deps.graph_service is mock_graph
        assert deps.database_service is mock_database
        assert deps.reranker_service is mock_reranker
    
    def test_instantiation_with_partial_services(self, mock_llm, mock_embedding):
        """Test creating container with some services."""
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding
        )
        
        assert deps.llm_service is mock_llm
        assert deps.embedding_service is mock_embedding
        assert deps.graph_service is None
        assert deps.database_service is None
        assert deps.reranker_service is None


class TestValidationLogic:
    """Tests for validate_for_strategy method."""
    
    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create mock LLM service."""
        return Mock(spec=ILLMService)
    
    @pytest.fixture
    def mock_embedding(self) -> Mock:
        """Create mock embedding service."""
        return Mock(spec=IEmbeddingService)
    
    @pytest.fixture
    def mock_graph(self) -> Mock:
        """Create mock graph service."""
        return Mock(spec=IGraphService)
    
    @pytest.fixture
    def mock_database(self) -> Mock:
        """Create mock database service."""
        return Mock(spec=IDatabaseService)
    
    @pytest.fixture
    def mock_reranker(self) -> Mock:
        """Create mock reranking service."""
        return Mock(spec=IRerankingService)
    
    def test_validation_with_all_required_services_present(
        self,
        mock_llm,
        mock_embedding
    ):
        """Test validation passes when all required services are present."""
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding
        )
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        is_valid, missing = deps.validate_for_strategy(required)
        
        assert is_valid is True
        assert missing == []
    
    def test_validation_with_some_services_missing(self, mock_llm):
        """Test validation fails when some services are missing."""
        deps = StrategyDependencies(llm_service=mock_llm)
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        is_valid, missing = deps.validate_for_strategy(required)
        
        assert is_valid is False
        assert ServiceDependency.EMBEDDING in missing
        assert len(missing) == 1
    
    def test_validation_with_empty_requirements(self):
        """Test validation passes with empty requirements."""
        deps = StrategyDependencies()
        
        required = set()
        is_valid, missing = deps.validate_for_strategy(required)
        
        assert is_valid is True
        assert missing == []
    
    def test_validation_with_all_services_missing(self):
        """Test validation fails when all required services are missing."""
        deps = StrategyDependencies()
        
        required = {
            ServiceDependency.LLM,
            ServiceDependency.EMBEDDING,
            ServiceDependency.GRAPH
        }
        is_valid, missing = deps.validate_for_strategy(required)
        
        assert is_valid is False
        assert ServiceDependency.LLM in missing
        assert ServiceDependency.EMBEDDING in missing
        assert ServiceDependency.GRAPH in missing
        assert len(missing) == 3
    
    def test_validation_with_extra_services_present(
        self,
        mock_llm,
        mock_embedding,
        mock_graph
    ):
        """Test validation passes when extra services are present."""
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding,
            graph_service=mock_graph
        )
        
        required = {ServiceDependency.LLM}
        is_valid, missing = deps.validate_for_strategy(required)
        
        assert is_valid is True
        assert missing == []
    
    def test_validation_for_each_service_type(
        self,
        mock_llm,
        mock_embedding,
        mock_graph,
        mock_database,
        mock_reranker
    ):
        """Test validation for each individual service type."""
        # Test LLM
        deps = StrategyDependencies(llm_service=mock_llm)
        is_valid, missing = deps.validate_for_strategy({ServiceDependency.LLM})
        assert is_valid is True
        
        # Test EMBEDDING
        deps = StrategyDependencies(embedding_service=mock_embedding)
        is_valid, missing = deps.validate_for_strategy({ServiceDependency.EMBEDDING})
        assert is_valid is True
        
        # Test GRAPH
        deps = StrategyDependencies(graph_service=mock_graph)
        is_valid, missing = deps.validate_for_strategy({ServiceDependency.GRAPH})
        assert is_valid is True
        
        # Test DATABASE
        deps = StrategyDependencies(database_service=mock_database)
        is_valid, missing = deps.validate_for_strategy({ServiceDependency.DATABASE})
        assert is_valid is True
        
        # Test RERANKER
        deps = StrategyDependencies(reranker_service=mock_reranker)
        is_valid, missing = deps.validate_for_strategy({ServiceDependency.RERANKER})
        assert is_valid is True


class TestErrorMessaging:
    """Tests for get_missing_services_message method."""
    
    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create mock LLM service."""
        return Mock(spec=ILLMService)
    
    @pytest.fixture
    def mock_embedding(self) -> Mock:
        """Create mock embedding service."""
        return Mock(spec=IEmbeddingService)
    
    def test_message_for_valid_container(self, mock_llm, mock_embedding):
        """Test message is empty when all required services are present."""
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding
        )
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        message = deps.get_missing_services_message(required)
        
        assert message == ""
    
    def test_message_for_single_missing_service(self, mock_llm):
        """Test message for single missing service."""
        deps = StrategyDependencies(llm_service=mock_llm)
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        message = deps.get_missing_services_message(required)
        
        assert "Missing required services:" in message
        assert "EMBEDDING" in message
    
    def test_message_for_multiple_missing_services(self):
        """Test message lists all missing services."""
        deps = StrategyDependencies()
        
        required = {
            ServiceDependency.LLM,
            ServiceDependency.EMBEDDING,
            ServiceDependency.GRAPH
        }
        message = deps.get_missing_services_message(required)
        
        assert "Missing required services:" in message
        assert "LLM" in message
        assert "EMBEDDING" in message
        assert "GRAPH" in message
    
    def test_message_format(self):
        """Test message format is user-friendly."""
        deps = StrategyDependencies()
        
        required = {ServiceDependency.DATABASE}
        message = deps.get_missing_services_message(required)
        
        # Should start with standard prefix
        assert message.startswith("Missing required services:")
        # Should contain service name
        assert "DATABASE" in message
    
    def test_message_with_empty_requirements(self):
        """Test message is empty with no requirements."""
        deps = StrategyDependencies()
        
        required = set()
        message = deps.get_missing_services_message(required)
        
        assert message == ""
