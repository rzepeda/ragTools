"""Unit tests for retrieval interface.

This module tests the retrieval strategy interface, context, and example
implementation to ensure proper dependency validation, capability declaration,
and retrieval functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from rag_factory.core.retrieval_interface import (
    RetrievalContext,
    IRetrievalStrategy,
    RerankingRetrieval,
)
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import (
    ServiceDependency,
    StrategyDependencies,
)
from rag_factory.strategies.base import Chunk


class TestRetrievalContext:
    """Test suite for RetrievalContext."""

    def test_initialization_with_database_service(self):
        """Test context initialization with database service."""
        db_service = Mock()
        context = RetrievalContext(database_service=db_service)

        assert context.database is db_service
        assert context.config == {}
        assert context.metrics == {}

    def test_initialization_with_config(self):
        """Test context initialization with configuration."""
        db_service = Mock()
        config = {"top_k": 10, "threshold": 0.7}
        context = RetrievalContext(database_service=db_service, config=config)

        assert context.database is db_service
        assert context.config == config
        assert context.config["top_k"] == 10
        assert context.config["threshold"] == 0.7

    def test_metrics_tracking(self):
        """Test metrics dictionary can be updated."""
        db_service = Mock()
        context = RetrievalContext(database_service=db_service)

        context.metrics["chunks_retrieved"] = 10
        context.metrics["latency_ms"] = 150.5

        assert context.metrics["chunks_retrieved"] == 10
        assert context.metrics["latency_ms"] == 150.5


class TestIRetrievalStrategy:
    """Test suite for IRetrievalStrategy interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that IRetrievalStrategy cannot be instantiated directly."""
        deps = StrategyDependencies(database_service=Mock())
        config = {}

        with pytest.raises(TypeError):
            IRetrievalStrategy(config, deps)

    def test_dependency_validation_on_init(self):
        """Test that dependencies are validated during initialization."""

        class TestStrategy(IRetrievalStrategy):
            def requires(self):
                return {IndexCapability.VECTORS}

            def requires_services(self):
                return {ServiceDependency.DATABASE, ServiceDependency.RERANKER}

            async def retrieve(self, query, context, top_k=10):
                return []

        # Missing RERANKER service
        deps = StrategyDependencies(database_service=Mock())
        config = {}

        with pytest.raises(ValueError, match="Missing required services: RERANKER"):
            TestStrategy(config, deps)

    def test_successful_initialization_with_all_dependencies(self):
        """Test successful initialization when all dependencies are present."""

        class TestStrategy(IRetrievalStrategy):
            def requires(self):
                return {IndexCapability.VECTORS}

            def requires_services(self):
                return {ServiceDependency.DATABASE}

            async def retrieve(self, query, context, top_k=10):
                return []

        deps = StrategyDependencies(database_service=Mock())
        config = {"top_k": 10}

        strategy = TestStrategy(config, deps)

        assert strategy.config == config
        assert strategy.deps is deps

    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented."""

        # Missing retrieve method
        class IncompleteStrategy(IRetrievalStrategy):
            def requires(self):
                return set()

            def requires_services(self):
                return set()

        with pytest.raises(TypeError):
            IncompleteStrategy({}, StrategyDependencies())


class TestRerankingRetrieval:
    """Test suite for RerankingRetrieval example implementation."""

    def test_requires_capabilities(self):
        """Test that RerankingRetrieval declares correct capabilities."""
        deps = StrategyDependencies(
            database_service=Mock(),
            reranker_service=Mock()
        )
        strategy = RerankingRetrieval({}, deps)

        required_caps = strategy.requires()

        assert IndexCapability.VECTORS in required_caps
        assert IndexCapability.CHUNKS in required_caps
        assert IndexCapability.DATABASE in required_caps
        assert len(required_caps) == 3

    def test_requires_services(self):
        """Test that RerankingRetrieval declares correct services."""
        deps = StrategyDependencies(
            database_service=Mock(),
            reranker_service=Mock()
        )
        strategy = RerankingRetrieval({}, deps)

        required_services = strategy.requires_services()

        assert ServiceDependency.DATABASE in required_services
        assert ServiceDependency.RERANKER in required_services
        assert len(required_services) == 2

    def test_initialization_validates_dependencies(self):
        """Test that initialization fails with missing dependencies."""
        # Missing RERANKER service
        deps = StrategyDependencies(database_service=Mock())

        with pytest.raises(ValueError, match="Missing required services: RERANKER"):
            RerankingRetrieval({}, deps)

    @pytest.mark.asyncio
    async def test_retrieve_with_results(self):
        """Test retrieve method with successful results."""
        # Setup mocks
        db_service = Mock()
        reranker_service = Mock()

        # Create mock chunks
        candidate_chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata={},
                score=0.8,
                source_id="doc1",
                chunk_id=f"chunk_{i}"
            )
            for i in range(50)
        ]

        reranked_chunks = candidate_chunks[:10]

        db_service.search_vectors = AsyncMock(return_value=candidate_chunks)
        reranker_service.rerank = AsyncMock(return_value=reranked_chunks)

        deps = StrategyDependencies(
            database_service=db_service,
            reranker_service=reranker_service
        )

        config = {"initial_k": 50}
        strategy = RerankingRetrieval(config, deps)

        # Create context
        context = RetrievalContext(database_service=db_service)

        # Execute retrieval
        results = await strategy.retrieve("test query", context, top_k=10)

        # Verify results
        assert len(results) == 10
        assert results == reranked_chunks

        # Verify database was called correctly
        db_service.search_vectors.assert_called_once_with(
            query="test query",
            top_k=50
        )

        # Verify reranker was called correctly
        reranker_service.rerank.assert_called_once_with(
            query="test query",
            chunks=candidate_chunks,
            top_k=10
        )

        # Verify metrics were updated
        assert context.metrics["initial_candidates"] == 50
        assert context.metrics["final_results"] == 10

    @pytest.mark.asyncio
    async def test_retrieve_with_no_candidates(self):
        """Test retrieve method when no candidates are found."""
        # Setup mocks
        db_service = Mock()
        reranker_service = Mock()

        db_service.search_vectors = AsyncMock(return_value=[])
        reranker_service.rerank = AsyncMock()

        deps = StrategyDependencies(
            database_service=db_service,
            reranker_service=reranker_service
        )

        strategy = RerankingRetrieval({}, deps)
        context = RetrievalContext(database_service=db_service)

        # Execute retrieval
        results = await strategy.retrieve("test query", context, top_k=10)

        # Verify results
        assert results == []

        # Verify reranker was NOT called (no candidates)
        reranker_service.rerank.assert_not_called()

        # Verify metrics
        assert context.metrics["initial_candidates"] == 0
        assert context.metrics["final_results"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_uses_default_initial_k(self):
        """Test that retrieve uses default initial_k when not configured."""
        # Setup mocks
        db_service = Mock()
        reranker_service = Mock()

        db_service.search_vectors = AsyncMock(return_value=[])
        reranker_service.rerank = AsyncMock(return_value=[])

        deps = StrategyDependencies(
            database_service=db_service,
            reranker_service=reranker_service
        )

        # No initial_k in config
        strategy = RerankingRetrieval({}, deps)
        context = RetrievalContext(database_service=db_service)

        # Execute retrieval with top_k=10
        await strategy.retrieve("test query", context, top_k=10)

        # Verify default initial_k = top_k * 5 = 50
        db_service.search_vectors.assert_called_once_with(
            query="test query",
            top_k=50
        )

    @pytest.mark.asyncio
    async def test_retrieve_uses_configured_initial_k(self):
        """Test that retrieve uses configured initial_k."""
        # Setup mocks
        db_service = Mock()
        reranker_service = Mock()

        db_service.search_vectors = AsyncMock(return_value=[])
        reranker_service.rerank = AsyncMock(return_value=[])

        deps = StrategyDependencies(
            database_service=db_service,
            reranker_service=reranker_service
        )

        # Configure initial_k
        config = {"initial_k": 100}
        strategy = RerankingRetrieval(config, deps)
        context = RetrievalContext(database_service=db_service)

        # Execute retrieval
        await strategy.retrieve("test query", context, top_k=10)

        # Verify configured initial_k was used
        db_service.search_vectors.assert_called_once_with(
            query="test query",
            top_k=100
        )
