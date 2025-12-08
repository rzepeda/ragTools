"""Integration tests for factory consistency checking."""

import logging
import pytest
from typing import Set
from unittest.mock import MagicMock

from rag_factory.core.capabilities import IndexCapability
from rag_factory.factory import RAGFactory
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies
from rag_factory.strategies.base import IRAGStrategy


# Mock strategy classes for integration testing
class ConsistentIndexingStrategy(IRAGStrategy):
    """Mock indexing strategy with consistent capabilities and services."""
    
    def __init__(self, config, dependencies):
        super().__init__(config, dependencies)
    
    def produces(self) -> Set[IndexCapability]:
        return {IndexCapability.VECTORS, IndexCapability.DATABASE}
    
    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
    
    def prepare_data(self, documents):
        pass
    
    def retrieve(self, query, top_k):
        pass
    
    async def aretrieve(self, query, top_k):
        pass
    
    def process_query(self, query, context):
        pass


class InconsistentIndexingStrategy(IRAGStrategy):
    """Mock indexing strategy with inconsistent capabilities and services."""
    
    def __init__(self, config, dependencies):
        super().__init__(config, dependencies)
    
    def produces(self) -> Set[IndexCapability]:
        return {IndexCapability.VECTORS, IndexCapability.GRAPH}
    
    def requires_services(self) -> Set[ServiceDependency]:
        # Intentionally missing EMBEDDING and GRAPH services
        return set()
    
    def prepare_data(self, documents):
        pass
    
    def retrieve(self, query, top_k):
        pass
    
    async def aretrieve(self, query, top_k):
        pass
    
    def process_query(self, query, context):
        pass


class ConsistentRetrievalStrategy(IRAGStrategy):
    """Mock retrieval strategy with consistent capabilities and services."""
    
    def __init__(self, config, dependencies):
        super().__init__(config, dependencies)
    
    def requires(self) -> Set[IndexCapability]:
        return {IndexCapability.VECTORS}
    
    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.DATABASE, ServiceDependency.LLM}
    
    def prepare_data(self, documents):
        pass
    
    def retrieve(self, query, top_k):
        pass
    
    async def aretrieve(self, query, top_k):
        pass
    
    def process_query(self, query, context):
        pass


class InconsistentRetrievalStrategy(IRAGStrategy):
    """Mock retrieval strategy with inconsistent capabilities and services."""
    
    def __init__(self, config, dependencies):
        super().__init__(config, dependencies)
    
    def requires(self) -> Set[IndexCapability]:
        return {IndexCapability.VECTORS, IndexCapability.GRAPH}
    
    def requires_services(self) -> Set[ServiceDependency]:
        # Intentionally missing DATABASE and GRAPH services
        return {ServiceDependency.LLM}
    
    def prepare_data(self, documents):
        pass
    
    def retrieve(self, query, top_k):
        pass
    
    async def aretrieve(self, query, top_k):
        pass
    
    def process_query(self, query, context):
        pass


class StrategyWithoutCapabilityMethods(IRAGStrategy):
    """Mock strategy without produces()/requires() methods (legacy)."""
    
    def __init__(self, config, dependencies):
        super().__init__(config, dependencies)
    
    def requires_services(self) -> Set[ServiceDependency]:
        return set()
    
    def prepare_data(self, documents):
        pass
    
    def retrieve(self, query, top_k):
        pass
    
    async def aretrieve(self, query, top_k):
        pass
    
    def process_query(self, query, context):
        pass


class TestFactoryConsistencyIntegration:
    """Integration tests for factory consistency checking."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear registry before each test
        RAGFactory.clear_registry()
        yield
        # Clear registry after each test
        RAGFactory.clear_registry()
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return StrategyDependencies(
            llm_service=MagicMock(),
            embedding_service=MagicMock(),
            graph_service=MagicMock(),
            database_service=MagicMock(),
        )
    
    def test_consistent_indexing_strategy_no_warnings(self, mock_services, caplog):
        """Test that consistent indexing strategy produces no warnings."""
        factory = RAGFactory(
            embedding_service=mock_services.embedding_service,
            database_service=mock_services.database_service,
        )
        factory.register_strategy("consistent_indexing", ConsistentIndexingStrategy)
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("consistent_indexing", {})
        
        # Should not have any warnings
        assert len(caplog.records) == 0
        assert strategy is not None
    
    def test_inconsistent_indexing_strategy_logs_warnings(self, mock_services, caplog):
        """Test that inconsistent indexing strategy logs warnings."""
        factory = RAGFactory()
        factory.register_strategy("inconsistent_indexing", InconsistentIndexingStrategy)
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("inconsistent_indexing", {})
        
        # Should have warnings for missing EMBEDDING and GRAPH services
        assert len(caplog.records) == 2
        warning_messages = [record.message for record in caplog.records]
        assert any("VECTORS" in msg and "EMBEDDING" in msg for msg in warning_messages)
        assert any("GRAPH" in msg for msg in warning_messages)
        
        # Strategy should still be created despite warnings
        assert strategy is not None
    
    def test_consistent_retrieval_strategy_no_warnings(self, mock_services, caplog):
        """Test that consistent retrieval strategy produces no warnings."""
        factory = RAGFactory(
            llm_service=mock_services.llm_service,
            database_service=mock_services.database_service,
        )
        factory.register_strategy("consistent_retrieval", ConsistentRetrievalStrategy)
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("consistent_retrieval", {})
        
        # Should not have any warnings
        assert len(caplog.records) == 0
        assert strategy is not None
    
    def test_inconsistent_retrieval_strategy_logs_warnings(self, mock_services, caplog):
        """Test that inconsistent retrieval strategy logs warnings."""
        factory = RAGFactory(
            llm_service=mock_services.llm_service,
        )
        factory.register_strategy("inconsistent_retrieval", InconsistentRetrievalStrategy)
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("inconsistent_retrieval", {})
        
        # Should have warnings for missing DATABASE and GRAPH services
        assert len(caplog.records) == 2
        warning_messages = [record.message for record in caplog.records]
        assert any("VECTORS" in msg and "DATABASE" in msg for msg in warning_messages)
        assert any("GRAPH" in msg for msg in warning_messages)
        
        # Strategy should still be created despite warnings
        assert strategy is not None
    
    def test_legacy_strategy_without_capability_methods(self, caplog):
        """Test that legacy strategies without capability methods don't cause errors."""
        factory = RAGFactory()
        factory.register_strategy("legacy", StrategyWithoutCapabilityMethods)
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("legacy", {})
        
        # Should not have any warnings (capability checking is skipped)
        assert len(caplog.records) == 0
        assert strategy is not None
    
    def test_warning_messages_include_strategy_name(self, caplog):
        """Test that warning messages include the strategy class name."""
        factory = RAGFactory()
        factory.register_strategy("test_strategy", InconsistentIndexingStrategy)
        
        with caplog.at_level(logging.WARNING):
            factory.create_strategy("test_strategy", {})
        
        # All warnings should include the strategy class name
        for record in caplog.records:
            assert "InconsistentIndexingStrategy" in record.message
    
    def test_warning_messages_include_emoji(self, caplog):
        """Test that warning messages include the warning emoji."""
        factory = RAGFactory()
        factory.register_strategy("test_strategy", InconsistentIndexingStrategy)
        
        with caplog.at_level(logging.WARNING):
            factory.create_strategy("test_strategy", {})
        
        # All warnings should include the warning emoji
        for record in caplog.records:
            assert "⚠️" in record.message
    
    def test_override_dependencies_checked(self, mock_services, caplog):
        """Test that consistency checking works with override dependencies."""
        factory = RAGFactory()  # No default services
        factory.register_strategy("consistent", ConsistentIndexingStrategy)
        
        # Override with proper services
        override_deps = StrategyDependencies(
            embedding_service=mock_services.embedding_service,
            database_service=mock_services.database_service,
        )
        
        with caplog.at_level(logging.WARNING):
            strategy = factory.create_strategy("consistent", {}, override_deps=override_deps)
        
        # Should not have warnings with proper override dependencies
        assert len(caplog.records) == 0
        assert strategy is not None
