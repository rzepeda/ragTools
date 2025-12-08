"""Unit tests for ConsistencyChecker."""

import pytest
from typing import Set

from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.services.consistency import ConsistencyChecker


# Mock strategy classes for testing
class MockIndexingStrategy:
    """Mock indexing strategy for testing."""
    
    def __init__(self, produces: Set[IndexCapability], requires_services: Set[ServiceDependency]):
        self._produces = produces
        self._requires_services = requires_services
    
    def produces(self) -> Set[IndexCapability]:
        return self._produces
    
    def requires_services(self) -> Set[ServiceDependency]:
        return self._requires_services


class MockRetrievalStrategy:
    """Mock retrieval strategy for testing."""
    
    def __init__(self, requires: Set[IndexCapability], requires_services: Set[ServiceDependency]):
        self._requires = requires
        self._requires_services = requires_services
    
    def requires(self) -> Set[IndexCapability]:
        return self._requires
    
    def requires_services(self) -> Set[ServiceDependency]:
        return self._requires_services


class TestConsistencyChecker:
    """Test suite for ConsistencyChecker."""
    
    @pytest.fixture
    def checker(self):
        """Create a ConsistencyChecker instance."""
        return ConsistencyChecker()
    
    # Indexing Strategy Tests - Consistent Cases
    
    def test_indexing_consistent_vectors(self, checker):
        """Test consistent indexing strategy with VECTORS capability."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.VECTORS, IndexCapability.DATABASE},
            requires_services={ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 0
    
    def test_indexing_consistent_graph(self, checker):
        """Test consistent indexing strategy with GRAPH capability."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.GRAPH, IndexCapability.DATABASE},
            requires_services={ServiceDependency.GRAPH, ServiceDependency.DATABASE}
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 0
    
    def test_indexing_consistent_in_memory(self, checker):
        """Test consistent indexing strategy with IN_MEMORY capability."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.IN_MEMORY, IndexCapability.CHUNKS},
            requires_services=set()  # No database service
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 0
    
    # Indexing Strategy Tests - Inconsistent Cases
    
    def test_indexing_vectors_without_embedding(self, checker):
        """Test warning when VECTORS capability lacks EMBEDDING service."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.VECTORS},
            requires_services=set()  # Missing EMBEDDING
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 1
        assert "VECTORS" in warnings[0]
        assert "EMBEDDING" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_indexing_graph_without_graph_service(self, checker):
        """Test warning when GRAPH capability lacks GRAPH service."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.GRAPH},
            requires_services=set()  # Missing GRAPH
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 1
        assert "GRAPH" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_indexing_database_without_database_service(self, checker):
        """Test warning when DATABASE capability lacks DATABASE service."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.DATABASE},
            requires_services=set()  # Missing DATABASE
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 1
        assert "DATABASE" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_indexing_in_memory_with_database(self, checker):
        """Test warning when IN_MEMORY capability has DATABASE service."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.IN_MEMORY},
            requires_services={ServiceDependency.DATABASE}  # Unusual combination
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 1
        assert "IN_MEMORY" in warnings[0]
        assert "DATABASE" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_indexing_multiple_inconsistencies(self, checker):
        """Test multiple warnings for multiple inconsistencies."""
        strategy = MockIndexingStrategy(
            produces={IndexCapability.VECTORS, IndexCapability.GRAPH},
            requires_services=set()  # Missing both EMBEDDING and GRAPH
        )
        warnings = checker.check_indexing_strategy(strategy)
        assert len(warnings) == 2
        assert any("VECTORS" in w and "EMBEDDING" in w for w in warnings)
        assert any("GRAPH" in w for w in warnings)
    
    # Retrieval Strategy Tests - Consistent Cases
    
    def test_retrieval_consistent_vectors(self, checker):
        """Test consistent retrieval strategy with VECTORS requirement."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.VECTORS},
            requires_services={ServiceDependency.DATABASE, ServiceDependency.LLM}
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 0
    
    def test_retrieval_consistent_graph(self, checker):
        """Test consistent retrieval strategy with GRAPH requirement."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.GRAPH},
            requires_services={ServiceDependency.GRAPH, ServiceDependency.LLM}
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 0
    
    def test_retrieval_consistent_keywords(self, checker):
        """Test consistent retrieval strategy with KEYWORDS requirement."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.KEYWORDS},
            requires_services={ServiceDependency.DATABASE, ServiceDependency.LLM}
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 0
    
    # Retrieval Strategy Tests - Inconsistent Cases
    
    def test_retrieval_vectors_without_database(self, checker):
        """Test warning when VECTORS requirement lacks DATABASE service."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.VECTORS},
            requires_services={ServiceDependency.LLM}  # Missing DATABASE
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 1
        assert "VECTORS" in warnings[0]
        assert "DATABASE" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_retrieval_graph_without_graph_service(self, checker):
        """Test warning when GRAPH requirement lacks GRAPH service."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.GRAPH},
            requires_services={ServiceDependency.LLM}  # Missing GRAPH
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 1
        assert "GRAPH" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_retrieval_keywords_without_database(self, checker):
        """Test warning when KEYWORDS requirement lacks DATABASE service."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.KEYWORDS},
            requires_services={ServiceDependency.LLM}  # Missing DATABASE
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 1
        assert "KEYWORDS" in warnings[0]
        assert "DATABASE" in warnings[0]
        assert "⚠️" in warnings[0]
    
    def test_retrieval_multiple_inconsistencies(self, checker):
        """Test multiple warnings for multiple inconsistencies."""
        strategy = MockRetrievalStrategy(
            requires={IndexCapability.VECTORS, IndexCapability.GRAPH},
            requires_services={ServiceDependency.LLM}  # Missing DATABASE and GRAPH
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 2
        assert any("VECTORS" in w and "DATABASE" in w for w in warnings)
        assert any("GRAPH" in w for w in warnings)
    
    def test_retrieval_empty_requirements(self, checker):
        """Test no warnings for strategy with no capability requirements."""
        strategy = MockRetrievalStrategy(
            requires=set(),
            requires_services={ServiceDependency.LLM}
        )
        warnings = checker.check_retrieval_strategy(strategy)
        assert len(warnings) == 0
