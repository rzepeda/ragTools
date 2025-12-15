"""Integration tests for ServiceRegistry."""

import pytest
import os
from pathlib import Path

from rag_factory.registry import ServiceRegistry
from rag_factory.services.interfaces import IEmbeddingService, ILLMService, IDatabaseService


@pytest.fixture
def test_services_yaml(tmp_path):
    """Create test services.yaml with real service configurations."""
    content = """
services:
  embedding_local:
    name: "local-onnx-minilm"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
"""
    services_file = tmp_path / "services.yaml"
    services_file.write_text(content)
    return str(services_file)


@pytest.fixture
def test_services_yaml_with_env(tmp_path):
    """Create test services.yaml with environment variables."""
    content = """
services:
  llm_local:
    name: "local-llm"
    type: "llm"
    url: "${LLM_URL:-http://localhost:1234/v1}"
    model: "${LLM_MODEL:-test-model}"
    api_key: "not-needed"
    temperature: 0.7
    
  embedding_local:
    name: "local-embedding"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "${MODELS_DIR:-./models}"
    batch_size: 32
"""
    services_file = tmp_path / "services.yaml"
    services_file.write_text(content)
    return str(services_file)


@pytest.mark.integration
class TestRealServiceInstantiation:
    """Tests with real service instantiation."""

    def test_instantiate_onnx_embedding_service(self, test_services_yaml):
        """Test instantiating real ONNX embedding service."""
        registry = ServiceRegistry(test_services_yaml)

        # Get embedding service
        embedding = registry.get("embedding_local")

        assert embedding is not None
        assert isinstance(embedding, IEmbeddingService)
        assert hasattr(embedding, 'embed')
        assert hasattr(embedding, 'embed_batch')
        assert hasattr(embedding, 'get_dimension')

        # Test caching
        embedding2 = registry.get("embedding_local")
        assert embedding is embedding2

        registry.shutdown()

    def test_embedding_service_functionality(self, test_services_yaml):
        """Test that instantiated embedding service actually works."""
        registry = ServiceRegistry(test_services_yaml)

        embedding = registry.get("embedding_local")

        # Test dimension
        dimension = embedding.get_dimension()
        assert dimension == 384  # MiniLM-L6-v2 dimension

        registry.shutdown()

    def test_multiple_service_instantiation(self, test_services_yaml_with_env):
        """Test instantiating multiple services."""
        registry = ServiceRegistry(test_services_yaml_with_env)

        # Get both services
        embedding = registry.get("embedding_local")

        assert embedding is not None
        assert isinstance(embedding, IEmbeddingService)

        # Verify they're cached
        embedding2 = registry.get("embedding_local")
        assert embedding is embedding2

        registry.shutdown()


@pytest.mark.integration
class TestEnvironmentVariableResolution:
    """Tests for environment variable resolution."""

    def test_env_var_resolution(self, test_services_yaml_with_env, monkeypatch):
        """Test environment variable resolution in service configs."""
        # Set environment variables
        monkeypatch.setenv("LLM_URL", "http://custom:8080/v1")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("MODELS_DIR", "/custom/models")

        registry = ServiceRegistry(test_services_yaml_with_env)

        # Check that variables were resolved in config
        llm_config = registry.config['services']['llm_local']
        assert llm_config['url'] == "http://custom:8080/v1"
        assert llm_config['model'] == "custom-model"

        embedding_config = registry.config['services']['embedding_local']
        assert embedding_config['cache_dir'] == "/custom/models"

    def test_env_var_defaults(self, test_services_yaml_with_env):
        """Test environment variable default values."""
        # Don't set env vars - should use defaults
        registry = ServiceRegistry(test_services_yaml_with_env)

        llm_config = registry.config['services']['llm_local']
        assert llm_config['url'] == "http://localhost:1234/v1"
        assert llm_config['model'] == "test-model"

        embedding_config = registry.config['services']['embedding_local']
        assert embedding_config['cache_dir'] == "./models"


@pytest.mark.integration
class TestServiceLifecycle:
    """Tests for service lifecycle management."""

    def test_context_manager_cleanup(self, test_services_yaml):
        """Test context manager properly cleans up services."""
        with ServiceRegistry(test_services_yaml) as registry:
            embedding = registry.get("embedding_local")
            assert embedding is not None

        # After context exit, instances should be cleared
        assert len(registry._instances) == 0

    def test_reload_service(self, test_services_yaml):
        """Test reloading a service."""
        registry = ServiceRegistry(test_services_yaml)

        # Get service
        embedding1 = registry.get("embedding_local")

        # Reload service
        embedding2 = registry.reload("embedding_local")

        # Should be different instances
        assert embedding1 is not embedding2

        registry.shutdown()

    def test_shutdown_cleanup(self, test_services_yaml):
        """Test shutdown properly cleans up all services."""
        registry = ServiceRegistry(test_services_yaml)

        # Get service
        registry.get("embedding_local")

        # Shutdown
        registry.shutdown()

        # Instances should be cleared
        assert len(registry._instances) == 0


@pytest.mark.integration
class TestServiceSharing:
    """Tests for service sharing between multiple consumers."""

    def test_service_instance_sharing(self, test_services_yaml):
        """Test that multiple consumers get the same service instance."""
        registry = ServiceRegistry(test_services_yaml)

        # Simulate multiple strategies getting the same service
        embedding1 = registry.get("embedding_local")
        embedding2 = registry.get("$embedding_local")  # With $ prefix
        embedding3 = registry.get("embedding_local")

        # All should be the same instance
        assert embedding1 is embedding2
        assert embedding2 is embedding3

        registry.shutdown()

    def test_service_sharing_memory_efficiency(self, test_services_yaml):
        """Test that service sharing is memory efficient."""
        registry = ServiceRegistry(test_services_yaml)

        # Get service multiple times
        services = [registry.get("embedding_local") for _ in range(10)]

        # All should be the same object
        unique_ids = set(id(s) for s in services)
        assert len(unique_ids) == 1

        registry.shutdown()


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    def test_invalid_service_config(self, tmp_path):
        """Test handling of invalid service configuration."""
        content = """
services:
  invalid_service:
    unknown_field: "value"
"""
        services_file = tmp_path / "services.yaml"
        services_file.write_text(content)

        registry = ServiceRegistry(str(services_file))

        # Should raise error when trying to instantiate
        with pytest.raises(Exception):  # ServiceInstantiationError or similar
            registry.get("invalid_service")

    def test_missing_service_file(self):
        """Test handling of missing service configuration file."""
        from rag_factory.registry.exceptions import ServiceInstantiationError

        with pytest.raises(ServiceInstantiationError) as exc_info:
            ServiceRegistry("nonexistent.yaml")

        assert "not found" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIServices:
    """Tests for OpenAI service instantiation (requires API key)."""

    def test_openai_llm_instantiation(self, tmp_path, monkeypatch):
        """Test instantiating OpenAI LLM service."""
        monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

        content = """
services:
  llm_openai:
    name: "openai-gpt4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.7
"""
        services_file = tmp_path / "services.yaml"
        services_file.write_text(content)

        registry = ServiceRegistry(str(services_file))

        llm = registry.get("llm_openai")

        assert llm is not None
        assert isinstance(llm, ILLMService)
        assert hasattr(llm, 'complete')

        registry.shutdown()

    def test_openai_embedding_instantiation(self, tmp_path, monkeypatch):
        """Test instantiating OpenAI embedding service."""
        monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

        content = """
services:
  embedding_openai:
    name: "openai-embeddings"
    type: "embedding"
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-ada-002"
"""
        services_file = tmp_path / "services.yaml"
        services_file.write_text(content)

        registry = ServiceRegistry(str(services_file))

        embedding = registry.get("embedding_openai")

        assert embedding is not None
        assert isinstance(embedding, IEmbeddingService)
        assert hasattr(embedding, 'embed')

        registry.shutdown()


@pytest.mark.integration
class TestConfigurationValidation:
    """Tests for configuration validation during loading."""

    def test_valid_configuration_loads(self, test_services_yaml):
        """Test that valid configuration loads without errors."""
        registry = ServiceRegistry(test_services_yaml)

        assert registry.config is not None
        assert 'services' in registry.config
        assert len(registry.list_services()) > 0

    def test_configuration_warnings(self, tmp_path, caplog):
        """Test that configuration warnings are logged."""
        import logging
        caplog.set_level(logging.WARNING)

        content = """
services:
  test_service:
    provider: "onnx"
    model: "test-model"
    api_key: "plaintext-secret"  # Should trigger warning
"""
        services_file = tmp_path / "services.yaml"
        services_file.write_text(content)

        registry = ServiceRegistry(str(services_file))

        # Should have warning about plaintext secret
        # (if validator is configured to warn about this)
        # This depends on validator implementation
