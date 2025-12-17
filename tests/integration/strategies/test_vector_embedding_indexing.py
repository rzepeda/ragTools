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
    # Return embeddings matching the number of input texts
    service.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    service.get_dimension.return_value = 3
    return service

@pytest.fixture
def mock_database_service():
    service = AsyncMock(spec=IDatabaseService)
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
            IndexCapability.CHUNKS,
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
        documents = [{"id": "doc1", "text": "This is a test document with some content to chunk."}]
        
        result = await vector_embedding_strategy.process(documents, indexing_context)
        
        # Verify embedding service was called
        mock_embedding_service.embed_batch.assert_called()
        
        # Verify embeddings were stored
        mock_database_service.store_chunks.assert_called_once()
        call_args = mock_database_service.store_chunks.call_args
        chunks = call_args.args[0]
        
        # Verify chunks have embeddings
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'embedding' in chunk
            assert chunk['embedding'] == [0.1, 0.2, 0.3]
        
        # Verify result
        assert isinstance(result, IndexingResult)
        assert result.capabilities == {IndexCapability.VECTORS, IndexCapability.CHUNKS, IndexCapability.DATABASE}
        assert result.metadata['total_embeddings'] == len(chunks)

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
            config={'batch_size': 1, 'chunk_size': 10},  # Small chunk size to create multiple chunks
            dependencies=strategy_deps
        )
        
        documents = [{"id": "doc1", "text": "This is a longer test document that will be split into multiple chunks for testing batching."}]
        
        await strategy.process(documents, indexing_context)
        
        # Should be called multiple times because batch_size=1
        assert mock_embedding_service.embed_batch.call_count >= 2

    @pytest.mark.asyncio
    async def test_empty_documents_handled_gracefully(
        self, 
        vector_embedding_strategy, 
        indexing_context, 
        mock_database_service
    ):
        """Test that empty documents are handled gracefully."""
        mock_database_service.get_chunks_for_documents.return_value = []
        documents = [{"id": "doc1", "text": ""}]  # Empty document
        
        result = await vector_embedding_strategy.process(documents, indexing_context)
        
        # Should return empty result, not raise error
        assert result.chunk_count == 0
        assert result.document_count == 1

    @pytest.mark.asyncio
    async def test_service_error(
        self, 
        vector_embedding_strategy, 
        indexing_context, 
        mock_embedding_service
    ):
        """Test handling of service errors."""
        mock_embedding_service.embed_batch.side_effect = Exception("Service failed")
        documents = [{"id": "doc1", "text": "Test document"}]
        
        with pytest.raises(Exception, match="Service failed"):
            await vector_embedding_strategy.process(documents, indexing_context)
