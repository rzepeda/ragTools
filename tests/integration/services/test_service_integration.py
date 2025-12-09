"""Integration tests for service interactions."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Mock asyncpg before importing service
sys.modules["asyncpg"] = MagicMock()

from rag_factory.services.embedding.service import EmbeddingService
from rag_factory.services.embedding.config import EmbeddingServiceConfig
from rag_factory.services.embedding.base import EmbeddingResult
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig
from rag_factory.services.llm.base import LLMResponse, Message, MessageRole
from rag_factory.services.database.postgres import PostgresqlDatabaseService

@pytest.fixture
def embedding_service():
    """Create embedding service."""
    config = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        enable_cache=False
    )
    with patch("rag_factory.services.embedding.service.OpenAIProvider") as mock_provider_cls:
        mock_provider = Mock()
        mock_provider.get_embeddings.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            dimensions=3,
            token_count=10,
            cost=0.0001,
            provider="openai",
            cached=[False]
        )
        mock_provider.get_dimensions.return_value = 3
        mock_provider.get_max_batch_size.return_value = 100
        mock_provider.get_model_name.return_value = "text-embedding-3-small"
        mock_provider_cls.return_value = mock_provider
        
        service = EmbeddingService(config)
        service.provider = mock_provider
        return service

@pytest.fixture
def llm_service():
    """Create LLM service."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-sonnet-4.5",
        enable_rate_limiting=False
    )
    with patch("rag_factory.services.llm.service.AnthropicProvider") as mock_provider_cls:
        mock_provider = Mock()
        mock_provider.complete.return_value = LLMResponse(
            content="Generated answer",
            model="claude-sonnet-4.5",
            provider="anthropic",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.0001,
            latency=0.5,
            metadata={}
        )
        mock_provider.count_tokens.return_value = 10
        mock_provider.get_model_name.return_value = "claude-sonnet-4.5"
        mock_provider_cls.return_value = mock_provider
        
        service = LLMService(config)
        service.provider = mock_provider
        return service

@pytest.fixture
def database_service():
    """Create database service."""
    with patch("asyncpg.create_pool") as mock_create_pool, \
         patch("rag_factory.services.database.postgres.ASYNCPG_AVAILABLE", True):
        pool = AsyncMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        mock_create_pool.return_value = pool
        
        service = PostgresqlDatabaseService(
            host="localhost",
            database="test_db"
        )
        service._pool = pool
        return service, conn

@pytest.mark.asyncio
async def test_rag_workflow(embedding_service, llm_service, database_service):
    """Test full RAG workflow integration."""
    db_service, db_conn = database_service
    
    # 1. Embed document
    text = "Important document content"
    embedding_result = embedding_service.embed([text])
    embedding = embedding_result.embeddings[0]
    
    assert len(embedding) == 3
    
    # 2. Store in database
    chunk = {
        "chunk_id": "doc1",
        "text": text,
        "embedding": embedding,
        "metadata": {"source": "test"}
    }
    await db_service.store_chunks([chunk])
    
    db_conn.execute.assert_called()
    
    # 3. Retrieve relevant chunks (mock search)
    db_conn.fetch.return_value = [
        {
            "chunk_id": "doc1",
            "text": text,
            "metadata": '{"source": "test"}',
            "similarity": 0.95
        }
    ]
    
    query = "What is important?"
    query_embedding = embedding_service.embed([query]).embeddings[0]
    results = await db_service.search_chunks(query_embedding)
    
    assert len(results) == 1
    assert results[0]["text"] == text
    
    # 4. Generate answer with LLM
    context = results[0]["text"]
    prompt = f"Context: {context}\nQuestion: {query}"
    messages = [Message(role=MessageRole.USER, content=prompt)]
    
    response = llm_service.complete(messages)
    
    assert response.content == "Generated answer"
    assert response.total_tokens > 0

@pytest.mark.asyncio
async def test_embedding_database_consistency(embedding_service, database_service):
    """Test consistency between embedding dimensions and database storage."""
    db_service, db_conn = database_service
    
    # Get embedding dimension
    dim = embedding_service.provider.get_dimensions()
    
    # Create chunk with correct dimension
    embedding = [0.1] * dim
    chunk = {
        "chunk_id": "1",
        "text": "test",
        "embedding": embedding
    }
    
    await db_service.store_chunks([chunk])
    
    # Verify database received correct dimension vector
    call_args = db_conn.execute.call_args
    embedding_str = call_args[0][3] # 4th argument is embedding_str
    stored_dim = embedding_str.count(",") + 1
    
    assert stored_dim == dim
