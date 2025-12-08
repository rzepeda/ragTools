import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
from rag_factory.core.indexing_interface import IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies
from rag_factory.services.interfaces import IEmbeddingService, IDatabaseService

@pytest.fixture
def mock_embedding_service():
    service = AsyncMock(spec=IEmbeddingService)
    service.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 2  # Mock embeddings
    service.get_dimension.return_value = 3
    return service

@pytest.fixture
def mock_database_service():
    service = AsyncMock(spec=IDatabaseService)
    # Mock chunks return
    service.get_chunks_for_documents.return_value = [
        {'id': 'chunk1', 'text': 'text1'},
        {'id': 'chunk2', 'text': 'text2'}
    ]
    return service

@pytest.fixture
def strategy_deps(mock_embedding_service, mock_database_service):
    return StrategyDependencies(
        embedding_service=mock_embedding_service,
        database_service=mock_database_service
    )

@pytest.fixture
def indexing_context(mock_database_service):
    return IndexingContext(
        database_service=mock_database_service,
        config={}
    )

@pytest.fixture
def vector_embedding_strategy(strategy_deps):
    strategy = VectorEmbeddingIndexing(
        config={'batch_size': 2},
        dependencies=strategy_deps
    )
    return strategy

class TestVectorEmbeddingIndexing:
    
    def test_capabilities_and_dependencies(self, vector_embedding_strategy):
        """Test that the strategy declares correct capabilities and dependencies."""
        assert vector_embedding_strategy.produces() == {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
        assert vector_embedding_strategy.requires_services() == {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }

    @pytest.mark.asyncio
    async def test_process_success(
        self, 
        vector_embedding_strategy, 
        indexing_context, 
        mock_embedding_service, 
        mock_database_service
    ):
        """Test successful processing of documents."""
        documents = [{"id": "doc1", "content": "content"}]
        
        result = await vector_embedding_strategy.process(documents, indexing_context)
        
        # Verify chunks were retrieved
        mock_database_service.get_chunks_for_documents.assert_called_once_with(["doc1"])
        
        # Verify embedding service was called
        mock_embedding_service.embed_batch.assert_called()
        
        # Verify embeddings were stored
        mock_database_service.store_chunks.assert_called_once()
        call_args = mock_database_service.store_chunks.call_args
        chunks = call_args.args[0]
        assert len(chunks) == 2
        assert chunks[0]['embedding'] == [0.1, 0.2, 0.3]
        assert chunks[1]['embedding'] == [0.1, 0.2, 0.3]
        
        # Verify result
        assert isinstance(result, IndexingResult)
        assert result.capabilities == {IndexCapability.VECTORS, IndexCapability.DATABASE}
        assert result.metadata['total_embeddings'] == 2

    @pytest.mark.asyncio
    async def test_process_batching(
        self, 
        strategy_deps, 
        indexing_context, 
        mock_embedding_service
    ):
        """Test that batching logic works correctly."""
        # Setup strategy with small batch size
        strategy = VectorEmbeddingIndexing(
            config={'batch_size': 1},
            dependencies=strategy_deps
        )
        
        documents = [{"id": "doc1", "content": "content"}]
        
        await strategy.process(documents, indexing_context)
        
        # Should be called twice because we have 2 chunks and batch_size=1
        assert mock_embedding_service.embed_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_no_chunks_error(
        self, 
        vector_embedding_strategy, 
        indexing_context, 
        mock_database_service
    ):
        """Test that ValueError is raised when no chunks are found."""
        mock_database_service.get_chunks_for_documents.return_value = []
        documents = [{"id": "doc1", "content": "content"}]
        
        with pytest.raises(ValueError, match="No chunks found"):
            await vector_embedding_strategy.process(documents, indexing_context)

    @pytest.mark.asyncio
    async def test_service_error(
        self, 
        vector_embedding_strategy, 
        indexing_context, 
        mock_embedding_service
    ):
        """Test handling of service errors."""
        mock_embedding_service.embed_batch.side_effect = Exception("Service failed")
        documents = [{"id": "doc1", "content": "content"}]
        
        with pytest.raises(Exception, match="Service failed"):
            await vector_embedding_strategy.process(documents, indexing_context)
