"""Unit tests for indexing interface and context.

This module tests the IndexingContext, IIndexingStrategy interface,
VectorEmbeddingIndexing example implementation, and validate_dependencies function.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Set, Dict, Any, List

from rag_factory.core.indexing_interface import (
    IndexingContext,
    IIndexingStrategy,
    VectorEmbeddingIndexing,
)
from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.services.dependencies import (
    ServiceDependency,
    StrategyDependencies,
    validate_dependencies,
)


class TestIndexingContext:
    """Tests for IndexingContext class."""
    
    def test_initialization_with_database_service(self):
        """Test context initializes with database service."""
        db_service = Mock()
        context = IndexingContext(database_service=db_service)
        
        assert context.database is db_service
        assert context.config == {}
        assert context.metrics == {}
    
    def test_initialization_with_config(self):
        """Test context initializes with custom config."""
        db_service = Mock()
        config = {"chunk_size": 512, "chunk_overlap": 50}
        context = IndexingContext(database_service=db_service, config=config)
        
        assert context.config == config
        assert context.config["chunk_size"] == 512
    
    def test_metrics_tracking(self):
        """Test metrics can be tracked in context."""
        db_service = Mock()
        context = IndexingContext(database_service=db_service)
        
        context.metrics["chunks_processed"] = 100
        context.metrics["duration_seconds"] = 12.5
        
        assert context.metrics["chunks_processed"] == 100
        assert context.metrics["duration_seconds"] == 12.5
    
    def test_config_defaults_to_empty_dict(self):
        """Test config defaults to empty dict when not provided."""
        db_service = Mock()
        context = IndexingContext(database_service=db_service)
        
        assert isinstance(context.config, dict)
        assert len(context.config) == 0


class TestIIndexingStrategy:
    """Tests for IIndexingStrategy interface."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that IIndexingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IIndexingStrategy({}, StrategyDependencies())
    
    def test_requires_produces_implementation(self):
        """Test that subclasses must implement produces method."""
        class IncompleteStrategy(IIndexingStrategy):
            def requires_services(self) -> Set[ServiceDependency]:
                return set()
            
            async def process(self, documents, context):
                pass
        
        # Should fail because produces() is not implemented
        with pytest.raises(TypeError):
            IncompleteStrategy({}, StrategyDependencies())
    
    def test_requires_requires_services_implementation(self):
        """Test that subclasses must implement requires_services method."""
        class IncompleteStrategy(IIndexingStrategy):
            def produces(self) -> Set[IndexCapability]:
                return set()
            
            async def process(self, documents, context):
                pass
        
        # Should fail because requires_services() is not implemented
        with pytest.raises(TypeError):
            IncompleteStrategy({}, StrategyDependencies())
    
    def test_requires_process_implementation(self):
        """Test that subclasses must implement process method."""
        class IncompleteStrategy(IIndexingStrategy):
            def produces(self) -> Set[IndexCapability]:
                return set()
            
            def requires_services(self) -> Set[ServiceDependency]:
                return set()
        
        # Should fail because process() is not implemented
        with pytest.raises(TypeError):
            IncompleteStrategy({}, StrategyDependencies())
    
    def test_complete_implementation_succeeds(self):
        """Test that complete implementation can be instantiated."""
        class CompleteStrategy(IIndexingStrategy):
            def produces(self) -> Set[IndexCapability]:
                return {IndexCapability.VECTORS}
            
            def requires_services(self) -> Set[ServiceDependency]:
                return set()
            
            async def process(self, documents, context):
                return IndexingResult(
                    capabilities=self.produces(),
                    metadata={},
                    document_count=0,
                    chunk_count=0
                )
        
        strategy = CompleteStrategy({}, StrategyDependencies())
        assert strategy is not None
        assert strategy.produces() == {IndexCapability.VECTORS}
    
    def test_dependency_validation_on_init(self):
        """Test that dependencies are validated on initialization."""
        class TestStrategy(IIndexingStrategy):
            def produces(self) -> Set[IndexCapability]:
                return {IndexCapability.VECTORS}
            
            def requires_services(self) -> Set[ServiceDependency]:
                return {ServiceDependency.EMBEDDING}
            
            async def process(self, documents, context):
                pass
        
        # Should fail - missing embedding service
        with pytest.raises(ValueError, match="Missing required services: EMBEDDING"):
            TestStrategy({}, StrategyDependencies())
        
        # Should succeed - embedding service provided
        deps = StrategyDependencies(embedding_service=Mock())
        strategy = TestStrategy({}, deps)
        assert strategy is not None


class TestVectorEmbeddingIndexing:
    """Tests for VectorEmbeddingIndexing example implementation."""
    
    def test_produces_correct_capabilities(self):
        """Test that strategy declares correct capabilities."""
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        capabilities = strategy.produces()
        
        assert IndexCapability.VECTORS in capabilities
        assert IndexCapability.CHUNKS in capabilities
        assert IndexCapability.DATABASE in capabilities
    
    def test_requires_correct_services(self):
        """Test that strategy declares correct service requirements."""
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        required = strategy.requires_services()
        
        assert ServiceDependency.EMBEDDING in required
        assert ServiceDependency.DATABASE in required
    
    def test_initialization_fails_without_embedding_service(self):
        """Test that initialization fails without embedding service."""
        database_service = Mock()
        deps = StrategyDependencies(database_service=database_service)
        
        with pytest.raises(ValueError, match="Missing required services: EMBEDDING"):
            VectorEmbeddingIndexing({}, deps)
    
    def test_initialization_fails_without_database_service(self):
        """Test that initialization fails without database service."""
        embedding_service = Mock()
        deps = StrategyDependencies(embedding_service=embedding_service)
        
        with pytest.raises(ValueError, match="Missing required services: DATABASE"):
            VectorEmbeddingIndexing({}, deps)
    
    def test_initialization_succeeds_with_all_services(self):
        """Test that initialization succeeds with all required services."""
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        assert strategy is not None
        assert strategy.deps.embedding_service is embedding_service
        assert strategy.deps.database_service is database_service
    
    @pytest.mark.asyncio
    async def test_process_chunks_documents(self):
        """Test that process method chunks documents."""
        embedding_service = Mock()
        embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        
        database_service = Mock()
        database_service.store_chunks = AsyncMock()
        
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        config = {"chunk_size": 10, "chunk_overlap": 2}
        strategy = VectorEmbeddingIndexing(config, deps)
        
        context = IndexingContext(database_service, config)
        documents = [{"text": "This is a test document for chunking"}]
        
        result = await strategy.process(documents, context)
        
        # Should create multiple chunks
        assert result.chunk_count > 1
        assert result.document_count == 1
    
    @pytest.mark.asyncio
    async def test_process_generates_embeddings(self):
        """Test that process generates embeddings for chunks."""
        embedding_service = Mock()
        embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        
        database_service = Mock()
        database_service.store_chunks = AsyncMock()
        
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        context = IndexingContext(database_service)
        documents = [{"text": "Short text"}]
        
        await strategy.process(documents, context)
        
        # Verify embed_batch was called
        embedding_service.embed_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_stores_chunks_in_database(self):
        """Test that process stores chunks in database."""
        embedding_service = Mock()
        embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        
        database_service = Mock()
        database_service.store_chunks = AsyncMock()
        
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        context = IndexingContext(database_service)
        documents = [{"text": "Test document"}]
        
        await strategy.process(documents, context)
        
        # Verify store_chunks was called
        database_service.store_chunks.assert_called_once()
        
        # Verify chunks have embeddings
        stored_chunks = database_service.store_chunks.call_args[0][0]
        assert all("embedding" in chunk for chunk in stored_chunks)
    
    @pytest.mark.asyncio
    async def test_process_updates_context_metrics(self):
        """Test that process updates context metrics."""
        embedding_service = Mock()
        embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        
        database_service = Mock()
        database_service.store_chunks = AsyncMock()
        
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        context = IndexingContext(database_service)
        documents = [{"text": "Test"}]
        
        await strategy.process(documents, context)
        
        assert "chunks_created" in context.metrics
        assert "documents_processed" in context.metrics
        assert context.metrics["documents_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_returns_correct_result(self):
        """Test that process returns correct IndexingResult."""
        embedding_service = Mock()
        embedding_service.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        
        database_service = Mock()
        database_service.store_chunks = AsyncMock()
        
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        context = IndexingContext(database_service)
        documents = [{"text": "Test"}]
        
        result = await strategy.process(documents, context)
        
        assert isinstance(result, IndexingResult)
        assert result.capabilities == strategy.produces()
        assert result.document_count == 1
        assert result.chunk_count > 0
        assert "chunk_size" in result.metadata
    
    def test_chunk_documents_with_default_config(self):
        """Test chunking with default configuration."""
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        documents = [{"text": "a" * 1000}]  # 1000 character document
        
        chunks = strategy._chunk_documents(documents, 512, 50)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should have required fields
        assert all("text" in chunk for chunk in chunks)
        assert all("document_id" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)
    
    def test_chunk_documents_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        strategy = VectorEmbeddingIndexing({}, deps)
        documents = [{
            "text": "Test document",
            "metadata": {"source": "test.pdf", "page": 1}
        }]
        
        chunks = strategy._chunk_documents(documents, 10, 2)
        
        # All chunks should have the same metadata
        assert all(chunk["metadata"] == {"source": "test.pdf", "page": 1} for chunk in chunks)


class TestValidateDependencies:
    """Tests for validate_dependencies function."""
    
    def test_validation_passes_with_all_services(self):
        """Test validation passes when all required services are present."""
        llm_service = Mock()
        embedding_service = Mock()
        deps = StrategyDependencies(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        
        # Should not raise
        validate_dependencies(deps, required)
    
    def test_validation_fails_with_missing_service(self):
        """Test validation fails when required service is missing."""
        llm_service = Mock()
        deps = StrategyDependencies(llm_service=llm_service)
        
        required = {ServiceDependency.LLM, ServiceDependency.EMBEDDING}
        
        with pytest.raises(ValueError, match="Missing required services: EMBEDDING"):
            validate_dependencies(deps, required)
    
    def test_validation_fails_with_multiple_missing_services(self):
        """Test validation fails with multiple missing services."""
        deps = StrategyDependencies()
        
        required = {
            ServiceDependency.LLM,
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }
        
        with pytest.raises(ValueError, match="Missing required services"):
            validate_dependencies(deps, required)
    
    def test_validation_passes_with_empty_requirements(self):
        """Test validation passes when no services are required."""
        deps = StrategyDependencies()
        required = set()
        
        # Should not raise
        validate_dependencies(deps, required)
    
    def test_validation_passes_with_extra_services(self):
        """Test validation passes when extra services are provided."""
        llm_service = Mock()
        embedding_service = Mock()
        database_service = Mock()
        deps = StrategyDependencies(
            llm_service=llm_service,
            embedding_service=embedding_service,
            database_service=database_service
        )
        
        required = {ServiceDependency.LLM}
        
        # Should not raise - extra services are fine
        validate_dependencies(deps, required)
