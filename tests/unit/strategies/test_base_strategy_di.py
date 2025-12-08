"""Unit tests for base strategy dependency injection."""

import pytest
from typing import Set, Dict, Any, List
from unittest.mock import Mock, MagicMock

from rag_factory.strategies.base import IRAGStrategy
from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
from rag_factory.services.interfaces import ILLMService, IEmbeddingService, IDatabaseService


class ConcreteTestStrategy(IRAGStrategy):
    """Concrete strategy for testing DI validation."""
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Requires LLM and Embedding services."""
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        """Stub implementation."""
        pass
    
    def retrieve(self, query: str, top_k: int):
        """Stub implementation."""
        return []
    
    async def aretrieve(self, query: str, top_k: int):
        """Stub implementation."""
        return []
    
    def process_query(self, query: str, context):
        """Stub implementation."""
        return "test answer"


class TestBaseStrategyDI:
    """Test suite for base strategy dependency injection."""
    
    def test_strategy_with_all_required_services(self):
        """Test that strategy initializes successfully with all required services."""
        # Arrange
        mock_llm = Mock(spec=ILLMService)
        mock_embedding = Mock(spec=IEmbeddingService)
        
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding
        )
        
        config = {"test_param": "value"}
        
        # Act
        strategy = ConcreteTestStrategy(config, deps)
        
        # Assert
        assert strategy.config == config
        assert strategy.deps == deps
        assert strategy.deps.llm_service == mock_llm
        assert strategy.deps.embedding_service == mock_embedding
    
    def test_strategy_fails_without_required_llm_service(self):
        """Test that strategy fails to initialize without required LLM service."""
        # Arrange
        mock_embedding = Mock(spec=IEmbeddingService)
        
        deps = StrategyDependencies(
            embedding_service=mock_embedding
            # Missing llm_service
        )
        
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestStrategy(config, deps)
        
        assert "ConcreteTestStrategy requires services" in str(exc_info.value)
        assert "LLM" in str(exc_info.value)
    
    def test_strategy_fails_without_required_embedding_service(self):
        """Test that strategy fails to initialize without required Embedding service."""
        # Arrange
        mock_llm = Mock(spec=ILLMService)
        
        deps = StrategyDependencies(
            llm_service=mock_llm
            # Missing embedding_service
        )
        
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestStrategy(config, deps)
        
        assert "ConcreteTestStrategy requires services" in str(exc_info.value)
        assert "EMBEDDING" in str(exc_info.value)
    
    def test_strategy_fails_with_no_services(self):
        """Test that strategy fails to initialize with no services."""
        # Arrange
        deps = StrategyDependencies()  # No services
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestStrategy(config, deps)
        
        error_msg = str(exc_info.value)
        assert "ConcreteTestStrategy requires services" in error_msg
        assert "LLM" in error_msg
        assert "EMBEDDING" in error_msg
    
    def test_error_message_is_clear_and_helpful(self):
        """Test that error messages clearly indicate missing services."""
        # Arrange
        deps = StrategyDependencies()
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestStrategy(config, deps)
        
        error_msg = str(exc_info.value)
        # Should mention the strategy class name
        assert "ConcreteTestStrategy" in error_msg
        # Should mention "requires services"
        assert "requires services" in error_msg
        # Should list the missing services
        assert "LLM" in error_msg and "EMBEDDING" in error_msg
    
    def test_strategy_with_extra_unused_services(self):
        """Test that strategy works fine with extra services it doesn't need."""
        # Arrange
        mock_llm = Mock(spec=ILLMService)
        mock_embedding = Mock(spec=IEmbeddingService)
        mock_database = Mock(spec=IDatabaseService)
        
        deps = StrategyDependencies(
            llm_service=mock_llm,
            embedding_service=mock_embedding,
            database_service=mock_database  # Extra service not required
        )
        
        config = {}
        
        # Act
        strategy = ConcreteTestStrategy(config, deps)
        
        # Assert
        assert strategy.deps.llm_service == mock_llm
        assert strategy.deps.embedding_service == mock_embedding
        assert strategy.deps.database_service == mock_database


class MinimalTestStrategy(IRAGStrategy):
    """Strategy that requires no services for testing."""
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Requires no services."""
        return set()
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        pass
    
    def retrieve(self, query: str, top_k: int):
        return []
    
    async def aretrieve(self, query: str, top_k: int):
        return []
    
    def process_query(self, query: str, context):
        return "answer"


class TestStrategyWithNoRequirements:
    """Test strategies that don't require any services."""
    
    def test_strategy_with_no_service_requirements(self):
        """Test that strategy with no requirements works with empty dependencies."""
        # Arrange
        deps = StrategyDependencies()  # No services
        config = {}
        
        # Act
        strategy = MinimalTestStrategy(config, deps)
        
        # Assert
        assert strategy.config == config
        assert strategy.deps == deps
    
    def test_strategy_with_no_requirements_accepts_services(self):
        """Test that strategy with no requirements still accepts services."""
        # Arrange
        mock_llm = Mock(spec=ILLMService)
        deps = StrategyDependencies(llm_service=mock_llm)
        config = {}
        
        # Act
        strategy = MinimalTestStrategy(config, deps)
        
        # Assert
        assert strategy.deps.llm_service == mock_llm


class DatabaseRequiringStrategy(IRAGStrategy):
    """Strategy that requires database service."""
    
    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.DATABASE}
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        pass
    
    def retrieve(self, query: str, top_k: int):
        return []
    
    async def aretrieve(self, query: str, top_k: int):
        return []
    
    def process_query(self, query: str, context):
        return "answer"


class TestDifferentServiceCombinations:
    """Test various service dependency combinations."""
    
    def test_database_only_strategy(self):
        """Test strategy that only requires database service."""
        # Arrange
        mock_db = Mock(spec=IDatabaseService)
        deps = StrategyDependencies(database_service=mock_db)
        config = {}
        
        # Act
        strategy = DatabaseRequiringStrategy(config, deps)
        
        # Assert
        assert strategy.deps.database_service == mock_db
    
    def test_database_only_strategy_fails_without_database(self):
        """Test that database-only strategy fails without database."""
        # Arrange
        mock_llm = Mock(spec=ILLMService)
        deps = StrategyDependencies(llm_service=mock_llm)  # Wrong service
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            DatabaseRequiringStrategy(config, deps)
        
        assert "DATABASE" in str(exc_info.value)
