"""Unit tests for ServiceFactory."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.registry.service_factory import ServiceFactory
from rag_factory.registry.exceptions import ServiceInstantiationError


@pytest.fixture
def factory():
    """Create ServiceFactory instance."""
    return ServiceFactory()


class TestServiceTypeDetection:
    """Tests for service type detection methods."""

    def test_is_llm_service(self, factory):
        """Test LLM service detection."""
        config = {"url": "http://localhost:1234/v1", "model": "test-model"}
        assert factory._is_llm_service(config) is True

    def test_is_llm_service_missing_url(self, factory):
        """Test LLM service detection with missing URL."""
        config = {"model": "test-model"}
        assert factory._is_llm_service(config) is False

    def test_is_llm_service_missing_model(self, factory):
        """Test LLM service detection with missing model."""
        config = {"url": "http://localhost:1234/v1"}
        assert factory._is_llm_service(config) is False

    def test_is_embedding_service(self, factory):
        """Test embedding service detection."""
        config = {"provider": "onnx", "model": "test-model"}
        assert factory._is_embedding_service(config) is True

    def test_is_embedding_service_missing_provider(self, factory):
        """Test embedding service detection with missing provider."""
        config = {"model": "test-model"}
        assert factory._is_embedding_service(config) is False

    def test_is_database_service_postgres(self, factory):
        """Test database service detection for PostgreSQL."""
        config = {"type": "postgres", "connection_string": "postgresql://..."}
        assert factory._is_database_service(config) is True

    def test_is_database_service_neo4j(self, factory):
        """Test database service detection for Neo4j."""
        config = {"type": "neo4j", "uri": "bolt://localhost:7687"}
        assert factory._is_database_service(config) is True

    def test_is_database_service_unknown_type(self, factory):
        """Test database service detection with unknown type."""
        config = {"type": "unknown"}
        assert factory._is_database_service(config) is False


class TestLLMServiceCreation:
    """Tests for LLM service creation."""

    def test_create_llm_service_openai(self, factory):
        """Test creating OpenAI LLM service."""
        config = {
            "name": "openai-llm",
            "url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4",
            "temperature": 0.8,
            "max_tokens": 2000
        }

        with patch('rag_factory.services.api.OpenAILLMService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_llm_service("llm_openai", config)

            assert service is mock_instance
            # OpenAILLMService only accepts api_key and model in __init__
            mock_class.assert_called_once_with(
                api_key="sk-test",
                model="gpt-4"
            )

    def test_create_llm_service_lm_studio(self, factory):
        """Test creating LM Studio LLM service raises error (not yet supported)."""
        config = {
            "name": "test-llm",
            "url": "http://localhost:1234/v1",
            "model": "test-model",
            "temperature": 0.7
        }

        # LM Studio is not yet supported, should raise error
        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory._create_llm_service("llm1", config)

        assert "not yet fully supported" in str(exc_info.value)

    def test_create_llm_service_with_defaults(self, factory):
        """Test creating LLM service with default values (LM Studio not supported)."""
        config = {
            "url": "http://localhost:1234/v1",
            "model": "test-model"
        }

        # LM Studio is not yet supported, should raise error
        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory._create_llm_service("llm1", config)

        assert "not yet fully supported" in str(exc_info.value)


class TestEmbeddingServiceCreation:
    """Tests for embedding service creation."""

    def test_create_embedding_service_onnx(self, factory):
        """Test creating ONNX embedding service."""
        config = {
            "name": "onnx-embed",
            "provider": "onnx",
            "model": "Xenova/all-MiniLM-L6-v2",
            "cache_dir": "./models",
            "batch_size": 32
        }

        with patch('rag_factory.services.onnx.ONNXEmbeddingService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_embedding_service("embed1", config)

            assert service is mock_instance
            mock_class.assert_called_once_with(
                model="Xenova/all-MiniLM-L6-v2",
                cache_dir="./models",
                max_batch_size=32
            )

    def test_create_embedding_service_onnx_with_defaults(self, factory):
        """Test creating ONNX embedding service with defaults."""
        config = {
            "provider": "onnx",
            "model": "Xenova/all-MiniLM-L6-v2"
        }

        with patch('rag_factory.services.onnx.ONNXEmbeddingService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_embedding_service("embed1", config)

            assert service is mock_instance
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['cache_dir'] == './models'
            assert call_kwargs['max_batch_size'] == 32

    def test_create_embedding_service_openai(self, factory):
        """Test creating OpenAI embedding service."""
        config = {
            "provider": "openai",
            "api_key": "sk-test",
            "model": "text-embedding-ada-002"
        }

        with patch('rag_factory.services.api.OpenAIEmbeddingService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_embedding_service("embed_openai", config)

            assert service is mock_instance
            mock_class.assert_called_once_with(
                api_key="sk-test",
                model="text-embedding-ada-002"
            )

    def test_create_embedding_service_cohere_not_implemented(self, factory):
        """Test that Cohere embedding service raises error (not implemented)."""
        config = {
            "provider": "cohere",
            "api_key": "test-key",
            "model": "embed-english-v3.0"
        }

        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory._create_embedding_service("embed_cohere", config)

        assert "not yet implemented" in str(exc_info.value)

    def test_create_embedding_service_unknown_provider(self, factory):
        """Test creating embedding service with unknown provider."""
        config = {
            "provider": "unknown",
            "model": "test-model"
        }

        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory._create_embedding_service("embed1", config)

        assert "Unknown embedding provider" in str(exc_info.value)


class TestDatabaseServiceCreation:
    """Tests for database service creation."""

    def test_create_database_service_postgres_with_connection_string(self, factory):
        """Test creating PostgreSQL database service with connection string."""
        config = {
            "name": "postgres-db",
            "type": "postgres",
            "connection_string": "postgresql://user:pass@localhost:5432/db",
            "pool_size": 10,
            "max_overflow": 20
        }

        with patch('rag_factory.services.database.PostgresqlDatabaseService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_database_service("db1", config)

            assert service is mock_instance
            mock_class.assert_called_once_with(
                connection_string="postgresql://user:pass@localhost:5432/db",
                pool_size=10,
                max_overflow=20
            )

    def test_create_database_service_postgres_with_components(self, factory):
        """Test creating PostgreSQL database service with connection components."""
        config = {
            "type": "postgres",
            "user": "testuser",
            "password": "testpass",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "pool_size": 5
        }

        with patch('rag_factory.services.database.PostgresqlDatabaseService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_database_service("db1", config)

            assert service is mock_instance
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['connection_string'] == "postgresql://testuser:testpass@localhost:5432/testdb"
            assert call_kwargs['pool_size'] == 5

    def test_create_database_service_postgres_with_defaults(self, factory):
        """Test creating PostgreSQL database service with default values."""
        config = {
            "type": "postgres",
            "user": "testuser",
            "password": "testpass",
            "host": "localhost",
            "database": "testdb"
        }

        with patch('rag_factory.services.database.PostgresqlDatabaseService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_database_service("db1", config)

            assert service is mock_instance
            call_kwargs = mock_class.call_args[1]
            # Should use default port 5432
            assert "5432" in call_kwargs['connection_string']
            # Should use default pool_size and max_overflow
            assert call_kwargs['pool_size'] == 10
            assert call_kwargs['max_overflow'] == 20

    def test_create_database_service_neo4j(self, factory):
        """Test creating Neo4j database service."""
        config = {
            "type": "neo4j",
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "testpass"
        }

        with patch('rag_factory.services.database.Neo4jGraphService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_database_service("graph1", config)

            assert service is mock_instance
            mock_class.assert_called_once_with(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="testpass"
            )

    def test_create_database_service_neo4j_with_defaults(self, factory):
        """Test creating Neo4j database service with default URI."""
        config = {
            "type": "neo4j",
            "host": "localhost",
            "port": 7687,
            "user": "neo4j",
            "password": "testpass"
        }

        with patch('rag_factory.services.database.Neo4jGraphService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            service = factory._create_database_service("graph1", config)

            assert service is mock_instance
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['uri'] == "bolt://localhost:7687"

    def test_create_database_service_unknown_type(self, factory):
        """Test creating database service with unknown type."""
        config = {
            "type": "unknown",
            "connection_string": "unknown://..."
        }

        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory._create_database_service("db1", config)

        assert "Unknown database type" in str(exc_info.value)


class TestServiceCreation:
    """Tests for main create_service method."""

    def test_create_service_llm(self, factory):
        """Test creating LLM service through main method."""
        config = {
            "url": "http://localhost:1234/v1",
            "model": "test-model"
        }

        with patch.object(factory, '_create_llm_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            service = factory.create_service("llm1", config)

            assert service is mock_service
            mock_create.assert_called_once_with("llm1", config)

    def test_create_service_embedding(self, factory):
        """Test creating embedding service through main method."""
        config = {
            "provider": "onnx",
            "model": "test-model"
        }

        with patch.object(factory, '_create_embedding_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            service = factory.create_service("embed1", config)

            assert service is mock_service
            mock_create.assert_called_once_with("embed1", config)

    def test_create_service_database(self, factory):
        """Test creating database service through main method."""
        config = {
            "type": "postgres",
            "connection_string": "postgresql://..."
        }

        with patch.object(factory, '_create_database_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            service = factory.create_service("db1", config)

            assert service is mock_service
            mock_create.assert_called_once_with("db1", config)

    def test_create_service_unknown_type(self, factory):
        """Test creating service with unknown type raises error."""
        config = {
            "unknown_key": "unknown_value"
        }

        with pytest.raises(ServiceInstantiationError) as exc_info:
            factory.create_service("unknown", config)

        assert "Cannot determine service type" in str(exc_info.value)
        assert "unknown" in str(exc_info.value)
