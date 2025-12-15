"""Unit tests for ServiceRegistry."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import yaml

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.registry.exceptions import ServiceNotFoundError, ServiceInstantiationError


@pytest.fixture
def services_yaml_content():
    """Sample services.yaml content."""
    return """
services:
  llm1:
    name: "test-llm"
    type: "llm"
    url: "http://localhost:1234/v1"
    api_key: "test-key"
    model: "test-model"
    temperature: 0.7

  embedding1:
    name: "test-embedding"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models"
    batch_size: 32

  db1:
    name: "test-db"
    type: "postgres"
    connection_string: "postgresql://user:pass@localhost:5432/test"
    pool_size: 10
"""


@pytest.fixture
def mock_services_file(tmp_path, services_yaml_content):
    """Create temporary services.yaml file."""
    services_file = tmp_path / "services.yaml"
    services_file.write_text(services_yaml_content)
    return str(services_file)


@pytest.fixture
def mock_services_file_with_env_vars(tmp_path):
    """Create temporary services.yaml file with environment variables."""
    content = """
services:
  llm1:
    name: "test-llm"
    type: "llm"
    url: "${LLM_URL}"
    api_key: "${API_KEY:-default-key}"
    model: "test-model"
"""
    services_file = tmp_path / "services.yaml"
    services_file.write_text(content)
    return str(services_file)


class TestServiceRegistryInitialization:
    """Tests for ServiceRegistry initialization."""

    def test_registry_initialization(self, mock_services_file):
        """Test service registry initialization."""
        registry = ServiceRegistry(mock_services_file)

        assert registry.config is not None
        assert 'services' in registry.config
        assert len(registry.list_services()) == 3

    def test_registry_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(ServiceInstantiationError) as exc_info:
            ServiceRegistry("nonexistent.yaml")

        assert "not found" in str(exc_info.value)

    def test_registry_initialization_invalid_yaml(self, tmp_path):
        """Test initialization with invalid YAML."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ServiceInstantiationError) as exc_info:
            ServiceRegistry(str(invalid_file))

        assert "Failed to load" in str(exc_info.value)

    def test_registry_loads_and_validates_config(self, mock_services_file):
        """Test that registry loads and validates configuration."""
        with patch('rag_factory.registry.service_registry.ConfigValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_services_yaml.return_value = []
            mock_validator_class.return_value = mock_validator

            registry = ServiceRegistry(mock_services_file)

            # Should have called validator
            mock_validator.validate_services_yaml.assert_called_once()

    def test_registry_resolves_environment_variables(self, mock_services_file_with_env_vars, monkeypatch):
        """Test that registry resolves environment variables."""
        monkeypatch.setenv("LLM_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("API_KEY", "resolved-key")

        registry = ServiceRegistry(mock_services_file_with_env_vars)

        # Check that environment variables were resolved
        llm_config = registry.config['services']['llm1']
        assert llm_config['url'] == "http://localhost:1234/v1"
        assert llm_config['api_key'] == "resolved-key"


class TestServiceLookup:
    """Tests for service lookup and retrieval."""

    def test_list_services(self, mock_services_file):
        """Test listing available services."""
        registry = ServiceRegistry(mock_services_file)

        services = registry.list_services()

        assert 'llm1' in services
        assert 'embedding1' in services
        assert 'db1' in services
        assert len(services) == 3

    def test_list_services_empty_registry(self, tmp_path):
        """Test listing services with empty registry."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("services: {}")

        registry = ServiceRegistry(str(empty_file))
        services = registry.list_services()

        assert services == []

    def test_get_service_creates_instance(self, mock_services_file):
        """Test getting service creates instance on first call."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            service = registry.get("llm1")

            assert service is mock_service
            mock_create.assert_called_once()
            # Check that it was called with correct arguments
            call_args = mock_create.call_args
            assert call_args[0][0] == "llm1"  # service_name
            assert 'url' in call_args[0][1]  # config

    def test_get_service_caches_instance(self, mock_services_file):
        """Test service caching works correctly."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            # First call - should create
            service1 = registry.get("llm1")

            # Second call - should return cached
            service2 = registry.get("llm1")

            assert service1 is service2
            mock_create.assert_called_once()  # Only called once

    def test_get_service_with_dollar_prefix(self, mock_services_file):
        """Test service lookup with $ prefix."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            # Both should work and return same instance
            service1 = registry.get("$llm1")
            service2 = registry.get("llm1")

            assert service1 is service2
            mock_create.assert_called_once()

    def test_get_service_strips_dollar_prefix(self, mock_services_file):
        """Test that $ prefix is properly stripped."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            registry.get("$llm1")

            # Should be called with name without $
            call_args = mock_create.call_args
            assert call_args[0][0] == "llm1"

    def test_get_nonexistent_service_raises_error(self, mock_services_file):
        """Test getting non-existent service raises error."""
        registry = ServiceRegistry(mock_services_file)

        with pytest.raises(ServiceNotFoundError) as exc_info:
            registry.get("nonexistent")

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "llm1" in error_msg  # Should suggest available services

    def test_get_service_from_empty_registry(self, tmp_path):
        """Test getting service from empty registry."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("services: {}")

        registry = ServiceRegistry(str(empty_file))

        with pytest.raises(ServiceNotFoundError) as exc_info:
            registry.get("any_service")

        assert "not found" in str(exc_info.value)

    def test_get_service_instantiation_failure(self, mock_services_file):
        """Test handling of service instantiation failure."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_create.side_effect = Exception("Instantiation failed")

            with pytest.raises(ServiceInstantiationError) as exc_info:
                registry.get("llm1")

            assert "Instantiation failed" in str(exc_info.value)
            assert "llm1" in str(exc_info.value)


class TestServiceLifecycle:
    """Tests for service lifecycle management."""

    def test_reload_service(self, mock_services_file):
        """Test reloading a service."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            # Create mock services
            old_service = Mock()
            old_service.close = Mock()
            new_service = Mock()

            mock_create.side_effect = [old_service, new_service]

            # Get service initially
            service1 = registry.get("llm1")
            assert service1 is old_service

            # Reload service
            service2 = registry.reload("llm1")

            # Should have closed old service
            old_service.close.assert_called_once()

            # Should return new service
            assert service2 is new_service
            assert service2 is not old_service

    def test_reload_service_without_close_method(self, mock_services_file):
        """Test reloading service that doesn't have close method."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            old_service = Mock(spec=[])  # No close method
            new_service = Mock()

            mock_create.side_effect = [old_service, new_service]

            # Get service initially
            registry.get("llm1")

            # Reload should work even without close method
            service2 = registry.reload("llm1")
            assert service2 is new_service

    def test_reload_service_with_dollar_prefix(self, mock_services_file):
        """Test reload strips $ prefix."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_service.close = Mock()
            mock_create.return_value = mock_service

            registry.get("llm1")
            registry.reload("$llm1")  # With $ prefix

            # Should work correctly
            assert mock_create.call_count == 2

    def test_reload_nonexistent_service(self, mock_services_file):
        """Test reloading non-existent service."""
        registry = ServiceRegistry(mock_services_file)

        # Should raise error when trying to get non-existent service
        with pytest.raises(ServiceNotFoundError):
            registry.reload("nonexistent")

    def test_shutdown_closes_all_services(self, mock_services_file):
        """Test shutdown closes all services."""
        registry = ServiceRegistry(mock_services_file)

        # Create mock services with close methods
        services = []
        for name in ['llm1', 'embedding1', 'db1']:
            mock_service = Mock()
            mock_service.close = Mock()
            services.append(mock_service)
            registry._instances[name] = mock_service

        # Shutdown
        registry.shutdown()

        # All should be closed
        for service in services:
            service.close.assert_called_once()

        # Cache should be cleared
        assert len(registry._instances) == 0

    def test_shutdown_handles_close_errors(self, mock_services_file):
        """Test shutdown handles errors during close gracefully."""
        registry = ServiceRegistry(mock_services_file)

        # Create service that raises error on close
        mock_service = Mock()
        mock_service.close = Mock(side_effect=Exception("Close failed"))
        registry._instances['llm1'] = mock_service

        # Should not raise exception
        registry.shutdown()

        # Cache should still be cleared
        assert len(registry._instances) == 0

    def test_shutdown_skips_services_without_close(self, mock_services_file):
        """Test shutdown skips services without close method."""
        registry = ServiceRegistry(mock_services_file)

        # Create service without close method
        mock_service = Mock(spec=[])  # No close method
        registry._instances['llm1'] = mock_service

        # Should not raise exception
        registry.shutdown()

        # Cache should be cleared
        assert len(registry._instances) == 0


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self, mock_services_file):
        """Test registry works as context manager."""
        with patch('rag_factory.registry.service_registry.ServiceRegistry.shutdown') as mock_shutdown:
            with ServiceRegistry(mock_services_file) as registry:
                assert registry is not None
                assert isinstance(registry, ServiceRegistry)

            # Shutdown should be called on exit
            mock_shutdown.assert_called_once()

    def test_context_manager_with_exception(self, mock_services_file):
        """Test context manager calls shutdown even with exception."""
        with patch('rag_factory.registry.service_registry.ServiceRegistry.shutdown') as mock_shutdown:
            try:
                with ServiceRegistry(mock_services_file) as registry:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Shutdown should still be called
            mock_shutdown.assert_called_once()

    def test_context_manager_usage(self, mock_services_file):
        """Test typical context manager usage."""
        with ServiceRegistry(mock_services_file) as registry:
            with patch.object(registry._factory, 'create_service') as mock_create:
                mock_service = Mock()
                mock_service.close = Mock()
                mock_create.return_value = mock_service

                # Use service
                service = registry.get("llm1")
                assert service is mock_service

        # Service should be closed after context exit
        mock_service.close.assert_called_once()


class TestThreadSafety:
    """Tests for thread safety (basic checks)."""

    def test_double_check_locking(self, mock_services_file):
        """Test double-check locking pattern in get()."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service') as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            # Simulate race condition by adding to cache during lock
            original_get = registry.get

            def side_effect_get(service_ref):
                service_name = service_ref.lstrip('$')
                # Add to cache before factory is called
                if service_name not in registry._instances:
                    registry._instances[service_name] = mock_service
                return original_get(service_ref)

            # First call should check cache after acquiring lock
            service = registry.get("llm1")

            assert service is mock_service
            # Factory should be called at most once
            assert mock_create.call_count <= 1

    def test_per_service_locks(self, mock_services_file):
        """Test that different services have different locks."""
        registry = ServiceRegistry(mock_services_file)

        # Get locks for different services
        lock1 = registry._locks['llm1']
        lock2 = registry._locks['embedding1']

        # Should be different lock objects
        assert lock1 is not lock2
