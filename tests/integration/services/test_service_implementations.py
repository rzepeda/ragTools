"""Integration tests for service implementations.

These tests verify that all service implementations correctly implement
their respective interfaces from Story 11.1.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IRerankingService,
    IGraphService,
    IDatabaseService,
)
from rag_factory.services.onnx import ONNXEmbeddingService
from rag_factory.services.api import (
    AnthropicLLMService,
    OpenAILLMService,
    OpenAIEmbeddingService,
    CohereRerankingService,
)
from rag_factory.services.database import (
    Neo4jGraphService,
    PostgresqlDatabaseService,
)
from rag_factory.services.local import CosineRerankingService


class TestONNXServices:
    """Tests for ONNX service implementations."""

    @pytest.mark.asyncio
    async def test_onnx_embedding_service_implements_interface(self):
        """Test that ONNXEmbeddingService implements IEmbeddingService."""
        # This test verifies interface compliance at runtime
        assert issubclass(ONNXEmbeddingService, IEmbeddingService)

    @pytest.mark.asyncio
    async def test_onnx_embedding_service_basic_functionality(self):
        """Test basic embedding functionality with mocked provider."""
        with patch('rag_factory.services.onnx.embedding.ONNXLocalProvider') as MockProvider:
            # Setup mock
            mock_provider = MockProvider.return_value
            mock_result = Mock()
            mock_result.embeddings = [[0.1, 0.2, 0.3]]
            mock_provider.get_embeddings.return_value = mock_result
            mock_provider.get_dimensions.return_value = 3

            # Create service
            service = ONNXEmbeddingService(model="test-model")

            # Test embed
            embedding = await service.embed("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 3

            # Test get_dimension
            dim = service.get_dimension()
            assert dim == 3


class TestAPIServices:
    """Tests for API service implementations."""

    @pytest.mark.asyncio
    async def test_anthropic_llm_service_implements_interface(self):
        """Test that AnthropicLLMService implements ILLMService."""
        assert issubclass(AnthropicLLMService, ILLMService)

    @pytest.mark.asyncio
    async def test_anthropic_llm_service_basic_functionality(self):
        """Test basic LLM functionality with mocked provider."""
        with patch('rag_factory.services.api.anthropic.AnthropicProvider') as MockProvider:
            # Setup mock
            mock_provider = MockProvider.return_value
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_provider.complete.return_value = mock_response

            # Create service
            service = AnthropicLLMService(api_key="test-key")

            # Test complete
            response = await service.complete("test prompt")
            assert isinstance(response, str)
            assert response == "Test response"

    @pytest.mark.asyncio
    async def test_openai_llm_service_implements_interface(self):
        """Test that OpenAILLMService implements ILLMService."""
        assert issubclass(OpenAILLMService, ILLMService)

    @pytest.mark.asyncio
    async def test_openai_embedding_service_implements_interface(self):
        """Test that OpenAIEmbeddingService implements IEmbeddingService."""
        assert issubclass(OpenAIEmbeddingService, IEmbeddingService)

    @pytest.mark.asyncio
    async def test_cohere_reranking_service_implements_interface(self):
        """Test that CohereRerankingService implements IRerankingService."""
        assert issubclass(CohereRerankingService, IRerankingService)

    @pytest.mark.asyncio
    async def test_cohere_reranking_service_basic_functionality(self):
        """Test basic reranking functionality with mocked client."""
        with patch('rag_factory.services.api.cohere.cohere') as mock_cohere:
            # Setup mock
            mock_client = Mock()
            mock_cohere.Client.return_value = mock_client

            mock_result = Mock()
            mock_result.index = 0
            mock_result.relevance_score = 0.95

            mock_response = Mock()
            mock_response.results = [mock_result]
            mock_client.rerank.return_value = mock_response

            # Create service
            service = CohereRerankingService(api_key="test-key")

            # Test rerank
            results = await service.rerank("query", ["doc1", "doc2"], top_k=1)
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0] == (0, 0.95)


class TestDatabaseServices:
    """Tests for database service implementations."""

    @pytest.mark.asyncio
    async def test_neo4j_graph_service_implements_interface(self):
        """Test that Neo4jGraphService implements IGraphService."""
        assert issubclass(Neo4jGraphService, IGraphService)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(Neo4jGraphService, '__init__'),
        reason="neo4j package not installed"
    )
    async def test_neo4j_graph_service_basic_functionality(self):
        """Test basic graph operations with mocked driver."""
        try:
            import neo4j  # noqa: F401
        except ImportError:
            pytest.skip("neo4j package not installed")

        with patch('neo4j.AsyncGraphDatabase') as MockDB:
            # Setup mocks
            mock_driver = AsyncMock()
            MockDB.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_driver.session.return_value.__aenter__.return_value = mock_session

            mock_result = AsyncMock()
            mock_record = {"id": "node_123"}
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result

            # Create service
            service = Neo4jGraphService(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password"
            )

            # Test create_node
            node_id = await service.create_node("Person", {"name": "Alice"})
            assert node_id == "node_123"

    @pytest.mark.asyncio
    async def test_postgresql_database_service_implements_interface(self):
        """Test that PostgresqlDatabaseService implements IDatabaseService."""
        assert issubclass(PostgresqlDatabaseService, IDatabaseService)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(PostgresqlDatabaseService, '__init__'),
        reason="asyncpg package not installed"
    )
    async def test_postgresql_database_service_basic_functionality(self):
        """Test basic database operations with mocked pool."""
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            pytest.skip("asyncpg package not installed")

        # Setup mocks
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        # Mock the async context manager for pool.acquire()
        # We need to create a proper async context manager mock
        class AsyncContextManagerMock:
            async def __aenter__(self):
                return mock_conn
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        # Use Mock (not AsyncMock) for acquire since it's not an async function
        # It just returns an async context manager
        mock_pool.acquire = Mock(side_effect=lambda: AsyncContextManagerMock())

        # Create service
        service = PostgresqlDatabaseService(
            host="localhost",
            database="test_db",
            user="postgres",
            password="password"
        )
        
        # Pre-set the pool to avoid connection attempts
        service._pool = mock_pool

        # Test store_chunks (just verify it doesn't crash)
        chunks = [{
            "chunk_id": "test_1",
            "text": "Test content",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        }]
        await service.store_chunks(chunks)

        # Verify execute was called
        assert mock_conn.execute.called


class TestLocalServices:
    """Tests for local service implementations."""

    @pytest.mark.asyncio
    async def test_cosine_reranking_service_implements_interface(self):
        """Test that CosineRerankingService implements IRerankingService."""
        assert issubclass(CosineRerankingService, IRerankingService)

    @pytest.mark.asyncio
    async def test_cosine_reranking_service_basic_functionality(self):
        """Test basic reranking functionality with mocked embedding service."""
        # Create mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed.return_value = [1.0, 0.0, 0.0]
        mock_embedding_service.embed_batch.return_value = [
            [1.0, 0.0, 0.0],  # Perfect match
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.7, 0.7, 0.0],  # Partial match
        ]

        # Create service
        service = CosineRerankingService(mock_embedding_service)

        # Test rerank
        results = await service.rerank(
            "query",
            ["doc1", "doc2", "doc3"],
            top_k=2
        )

        assert isinstance(results, list)
        assert len(results) == 2
        # First result should be the perfect match (index 0)
        assert results[0][0] == 0
        # Scores should be in descending order
        assert results[0][1] >= results[1][1]


class TestInterfaceCompliance:
    """Tests to verify all services implement their interfaces correctly."""

    def test_all_llm_services_implement_interface(self):
        """Verify all LLM services implement ILLMService."""
        llm_services = [AnthropicLLMService, OpenAILLMService]
        for service_class in llm_services:
            assert issubclass(service_class, ILLMService), \
                f"{service_class.__name__} does not implement ILLMService"

    def test_all_embedding_services_implement_interface(self):
        """Verify all embedding services implement IEmbeddingService."""
        embedding_services = [ONNXEmbeddingService, OpenAIEmbeddingService]
        for service_class in embedding_services:
            assert issubclass(service_class, IEmbeddingService), \
                f"{service_class.__name__} does not implement IEmbeddingService"

    def test_all_reranking_services_implement_interface(self):
        """Verify all reranking services implement IRerankingService."""
        reranking_services = [CohereRerankingService, CosineRerankingService]
        for service_class in reranking_services:
            assert issubclass(service_class, IRerankingService), \
                f"{service_class.__name__} does not implement IRerankingService"

    def test_all_graph_services_implement_interface(self):
        """Verify all graph services implement IGraphService."""
        graph_services = [Neo4jGraphService]
        for service_class in graph_services:
            assert issubclass(service_class, IGraphService), \
                f"{service_class.__name__} does not implement IGraphService"

    def test_all_database_services_implement_interface(self):
        """Verify all database services implement IDatabaseService."""
        database_services = [PostgresqlDatabaseService]
        for service_class in database_services:
            assert issubclass(service_class, IDatabaseService), \
                f"{service_class.__name__} does not implement IDatabaseService"
