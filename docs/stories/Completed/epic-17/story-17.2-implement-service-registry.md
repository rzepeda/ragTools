# Story 17.2: Implement Service Registry

**Story ID:** 17.2
**Epic:** Epic 17 - Strategy Pair Configuration System
**Story Points:** 13
**Priority:** High
**Dependencies:** Story 17.1 (Configuration Schema), Epic 11 (Dependency Injection), Epic 16 (Database Consolidation)

---

## User Story

**As a** developer
**I want** a service registry that instantiates and caches service instances
**So that** multiple strategies can share the same service instances efficiently

---

## Detailed Requirements

### Functional Requirements

1. **Service Registry Core**
   - Load service configurations from services.yaml
   - Parse and validate service definitions
   - Maintain registry of available services
   - Provide service lookup by name
   - List all available services
   - Reload service configurations dynamically
   - Support hot-reloading for development

2. **Service Instantiation**
   - Lazy instantiation (create services only when first requested)
   - Factory pattern for service creation based on configuration
   - Support LLM service instantiation (LM Studio, OpenAI, etc.)
   - Support embedding service instantiation (ONNX, OpenAI, Cohere)
   - Support database service instantiation (PostgreSQL, Neo4j)
   - Proper error handling for instantiation failures
   - Validation of service configuration before instantiation

3. **Service Caching (Singleton Pattern)**
   - Cache instantiated services (one instance per service definition)
   - Thread-safe instance creation with locks
   - Prevent duplicate instantiations in concurrent scenarios
   - Efficient lookup of cached instances
   - Memory-efficient caching strategy
   - Detect and handle stale cache entries

4. **Environment Variable Resolution**
   - Resolve ${VAR} syntax in service configurations
   - Apply resolution before service instantiation
   - Support all resolution patterns (${VAR}, ${VAR:-default}, ${VAR:?error})
   - Cache resolved configurations
   - Re-resolve on configuration reload
   - Secure handling of secrets (no logging)

5. **Service Lifecycle Management**
   - Initialize services on first access
   - Properly close/cleanup services on shutdown
   - Support service reloading (close old, create new)
   - Handle service failures gracefully
   - Resource cleanup (connections, file handles, etc.)
   - Health checking for services

6. **Integration with Existing Services**
   - Use existing ILLMService, IEmbeddingService, IDatabaseService interfaces
   - Use existing service implementations (ONNXEmbeddingService, etc.)
   - No changes required to existing service code
   - Backward compatible with manual service instantiation
   - Bridge between configuration and code

### Non-Functional Requirements

1. **Performance**
   - Service lookup <10ms (cached)
   - First-time instantiation acceptable latency (<2s for model loading)
   - Lock contention minimal (per-service locks, not global)
   - Configuration parsing <100ms
   - Memory overhead <50MB for registry itself

2. **Thread Safety**
   - Thread-safe service instantiation
   - Thread-safe cache access
   - No race conditions in concurrent service creation
   - Proper lock granularity (avoid global locks)
   - Deadlock prevention

3. **Reliability**
   - Handle missing service configurations gracefully
   - Clear error messages for instantiation failures
   - Fallback mechanisms for failed services
   - No cascading failures (one bad service doesn't break registry)
   - Proper exception handling and propagation

4. **Observability**
   - Log service instantiations (timing, success/failure)
   - Track service usage (which services accessed)
   - Monitor cache hit rates
   - Diagnostic tools for debugging service issues
   - Performance metrics (instantiation time, cache size)

5. **Maintainability**
   - Clear separation of concerns (loading, validation, instantiation, caching)
   - Easy to add support for new service types
   - Well-documented service factory methods
   - Configuration-driven extensibility
   - Unit-testable components

---

## Acceptance Criteria

### AC1: Service Registry Loading
- [ ] ServiceRegistry class implemented
- [ ] Loads services.yaml configuration
- [ ] Validates configuration with ConfigValidator from Story 17.1
- [ ] Parses all service types (LLM, embedding, database)
- [ ] Handles malformed YAML gracefully
- [ ] Provides clear error messages for invalid configurations

### AC2: Service Instantiation
- [ ] Factory method creates LLM services (LM Studio, OpenAI)
- [ ] Factory method creates embedding services (ONNX, OpenAI, Cohere)
- [ ] Factory method creates database services (PostgreSQL)
- [ ] Uses existing service implementations from Epic 11
- [ ] Validates service configuration before instantiation
- [ ] Clear error messages for instantiation failures

### AC3: Service Caching
- [ ] Services cached after first instantiation
- [ ] Same instance returned for subsequent get() calls
- [ ] Thread-safe caching with per-service locks
- [ ] No duplicate instantiations in concurrent scenarios
- [ ] Cache invalidation on reload()
- [ ] Memory-efficient cache implementation

### AC4: Environment Variable Resolution
- [ ] ${VAR} syntax resolved in configurations
- [ ] ${VAR:-default} and ${VAR:?error} supported
- [ ] Resolution happens before instantiation
- [ ] Secrets not logged or exposed
- [ ] Resolution errors provide clear messages
- [ ] Re-resolution works on configuration reload

### AC5: Service Lifecycle
- [ ] Services instantiated lazily (on first get())
- [ ] shutdown() method closes all services
- [ ] reload(service_name) recreates specific service
- [ ] Old service instances properly cleaned up
- [ ] Resource leaks prevented (connections, files, etc.)
- [ ] Service failures don't crash registry

### AC6: Service Lookup
- [ ] get(service_name) returns service instance
- [ ] Handles "$service_name" and "service_name" formats
- [ ] Clear error for non-existent services
- [ ] list_services() returns all available services
- [ ] Fast lookup performance (<10ms cached)
- [ ] Supports service reference syntax from configurations

### AC7: Testing
- [ ] Unit tests for service loading (>90% coverage)
- [ ] Unit tests for each service type instantiation
- [ ] Unit tests for caching behavior
- [ ] Unit tests for thread safety
- [ ] Integration tests with real services
- [ ] Performance benchmarks for lookup and instantiation

---

## Technical Specifications

### File Structure
```
rag_factory/
├── registry/
│   ├── __init__.py
│   ├── service_registry.py       # ServiceRegistry class
│   ├── service_factory.py        # Service creation logic
│   └── exceptions.py              # Custom exceptions

tests/
├── unit/
│   └── registry/
│       ├── test_service_registry.py
│       ├── test_service_factory.py
│       └── test_threading.py
├── integration/
│   └── registry/
│       └── test_registry_integration.py
```

### Dependencies
```python
# requirements.txt - using existing dependencies
# threading (built-in)
# yaml (from Story 17.1)
```

### ServiceRegistry Implementation
```python
# rag_factory/registry/service_registry.py
from threading import Lock
from collections import defaultdict
from typing import Dict, Any, Optional
import logging
import yaml

from rag_factory.config.validator import ConfigValidator
from rag_factory.config.env_resolver import EnvResolver
from .service_factory import ServiceFactory
from .exceptions import ServiceNotFoundError, ServiceInstantiationError

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Central registry for service definitions and instances.

    Loads service configurations from YAML and creates/caches service instances.
    Multiple strategies can share the same service instance for efficiency.
    """

    def __init__(self, config_path: str = "config/services.yaml"):
        """
        Initialize service registry from configuration file.

        Args:
            config_path: Path to services.yaml configuration
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._instances: Dict[str, Any] = {}  # service_name -> instance
        self._locks = defaultdict(Lock)  # service_name -> lock
        self._validator = ConfigValidator()
        self._factory = ServiceFactory()

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate services.yaml configuration."""
        logger.info(f"Loading service registry from: {self.config_path}")

        try:
            # Load YAML
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            # Validate schema
            warnings = self._validator.validate_services_yaml(
                raw_config,
                file_path=self.config_path
            )

            # Print warnings
            for warning in warnings:
                logger.warning(warning)

            # Resolve environment variables
            self.config = EnvResolver.resolve(raw_config)

            logger.info(
                f"Service registry loaded: "
                f"{len(self.config.get('services', {}))} services available"
            )

        except FileNotFoundError:
            raise ServiceInstantiationError(
                f"Service registry configuration not found: {self.config_path}"
            )
        except Exception as e:
            raise ServiceInstantiationError(
                f"Failed to load service registry: {e}"
            )

    def get(self, service_ref: str) -> Any:
        """
        Get or create a service instance.

        Args:
            service_ref: Service reference like "$llm1" or "llm1"

        Returns:
            Service instance implementing appropriate interface

        Raises:
            ServiceNotFoundError: If service not found in registry
            ServiceInstantiationError: If service creation fails
        """
        # Strip $ prefix if present
        service_name = service_ref.lstrip('$')

        # Return cached instance if exists
        if service_name in self._instances:
            logger.debug(f"Service '{service_name}' returned from cache")
            return self._instances[service_name]

        # Thread-safe instantiation
        with self._locks[service_name]:
            # Double-check after acquiring lock
            if service_name in self._instances:
                return self._instances[service_name]

            # Validate service exists
            if 'services' not in self.config:
                raise ServiceNotFoundError(
                    f"No services defined in registry configuration"
                )

            if service_name not in self.config['services']:
                available = list(self.config['services'].keys())
                raise ServiceNotFoundError(
                    f"Service '{service_name}' not found in registry. "
                    f"Available services: {available}"
                )

            # Get service configuration
            service_config = self.config['services'][service_name]

            # Create service instance
            logger.info(f"Instantiating service: {service_name}")
            start_time = time.time()

            try:
                service_instance = self._factory.create_service(
                    service_name,
                    service_config
                )
                instantiation_time = time.time() - start_time

                logger.info(
                    f"Service '{service_name}' instantiated successfully "
                    f"in {instantiation_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Failed to instantiate service '{service_name}': {e}")
                raise ServiceInstantiationError(
                    f"Service instantiation failed for '{service_name}': {e}"
                )

            # Cache and return
            self._instances[service_name] = service_instance
            return service_instance

    def list_services(self) -> list[str]:
        """
        List all available service names.

        Returns:
            List of service names from configuration
        """
        return list(self.config.get('services', {}).keys())

    def reload(self, service_name: str) -> Any:
        """
        Force reload a service (useful after config changes).

        Closes old instance if it has a close() method, removes from cache,
        and returns newly instantiated service.

        Args:
            service_name: Service to reload (without $ prefix)

        Returns:
            New service instance
        """
        service_name = service_name.lstrip('$')

        logger.info(f"Reloading service: {service_name}")

        # Clean up old instance
        if service_name in self._instances:
            old_instance = self._instances[service_name]

            # Try to close gracefully
            if hasattr(old_instance, 'close'):
                try:
                    old_instance.close()
                    logger.debug(f"Closed old instance of '{service_name}'")
                except Exception as e:
                    logger.warning(
                        f"Error closing old service instance '{service_name}': {e}"
                    )

            # Remove from cache
            del self._instances[service_name]

        # Reload configuration
        self._load_config()

        # Next get() will create new instance
        return self.get(service_name)

    def shutdown(self) -> None:
        """
        Close all service instances and cleanup resources.

        Calls close() method on all services that support it.
        """
        logger.info("Shutting down service registry")

        for service_name, instance in self._instances.items():
            if hasattr(instance, 'close'):
                try:
                    instance.close()
                    logger.debug(f"Closed service: {service_name}")
                except Exception as e:
                    logger.warning(
                        f"Error closing service '{service_name}': {e}"
                    )

        self._instances.clear()
        logger.info("Service registry shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
        return False
```

### Service Factory Implementation
```python
# rag_factory/registry/service_factory.py
from typing import Any, Dict
import logging

# Import existing service interfaces from Epic 11
from rag_factory.services import (
    ILLMService,
    IEmbeddingService,
    IDatabaseService
)

# Import existing service implementations
from rag_factory.services.llm import LMStudioLLMService, OpenAILLMService
from rag_factory.services.embedding import ONNXEmbeddingService
from rag_factory.services.database import PostgresqlDatabaseService

from .exceptions import ServiceInstantiationError

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating service instances from configurations.

    Uses existing service implementations from Epic 11.
    """

    def create_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """
        Factory method to create service instances based on configuration.

        Args:
            service_name: Name of service (for logging)
            config: Service configuration dictionary

        Returns:
            Service instance implementing appropriate interface

        Raises:
            ServiceInstantiationError: If service type cannot be determined
        """
        # Determine service type based on configuration keys
        if self._is_llm_service(config):
            return self._create_llm_service(service_name, config)
        elif self._is_embedding_service(config):
            return self._create_embedding_service(service_name, config)
        elif self._is_database_service(config):
            return self._create_database_service(service_name, config)
        else:
            raise ServiceInstantiationError(
                f"Cannot determine service type for '{service_name}'. "
                f"Configuration: {config}"
            )

    def _is_llm_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents an LLM service."""
        return 'url' in config and 'model' in config

    def _is_embedding_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents an embedding service."""
        return 'provider' in config

    def _is_database_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents a database service."""
        return 'type' in config and config['type'] in ['postgres', 'neo4j', 'mongodb']

    def _create_llm_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ILLMService:
        """Create LLM service from configuration."""
        logger.debug(f"Creating LLM service: {service_name}")

        url = config['url']

        # Detect provider based on URL
        if 'openai.com' in url or 'api.openai.com' in url:
            # OpenAI LLM service
            return OpenAILLMService(
                api_key=config.get('api_key'),
                model=config['model'],
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens')
            )
        else:
            # LM Studio or other OpenAI-compatible service
            return LMStudioLLMService(
                base_url=url,
                api_key=config.get('api_key', 'not-needed'),
                model=config['model'],
                temperature=config.get('temperature', 0.7),
                timeout=config.get('timeout', 30)
            )

    def _create_embedding_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> IEmbeddingService:
        """Create embedding service from configuration."""
        logger.debug(f"Creating embedding service: {service_name}")

        provider = config['provider']

        if provider == 'onnx':
            # ONNX local embedding service
            return ONNXEmbeddingService(
                model_name=config['model'],
                cache_dir=config.get('cache_dir', './models'),
                batch_size=config.get('batch_size', 32)
            )
        elif provider == 'openai':
            # OpenAI embedding service
            from rag_factory.services.embedding import OpenAIEmbeddingService
            return OpenAIEmbeddingService(
                api_key=config['api_key'],
                model=config['model']
            )
        elif provider == 'cohere':
            # Cohere embedding service
            from rag_factory.services.embedding import CohereEmbeddingService
            return CohereEmbeddingService(
                api_key=config['api_key'],
                model=config['model']
            )
        else:
            raise ServiceInstantiationError(
                f"Unknown embedding provider: {provider}"
            )

    def _create_database_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> IDatabaseService:
        """Create database service from configuration."""
        logger.debug(f"Creating database service: {service_name}")

        db_type = config['type']

        if db_type == 'postgres':
            # PostgreSQL database service (from Epic 16)
            if 'connection_string' in config:
                conn_str = config['connection_string']
            else:
                # Build connection string from components
                conn_str = (
                    f"postgresql://{config['user']}:{config['password']}"
                    f"@{config['host']}:{config.get('port', 5432)}"
                    f"/{config['database']}"
                )

            return PostgresqlDatabaseService(
                connection_string=conn_str,
                pool_size=config.get('pool_size', 10),
                max_overflow=config.get('max_overflow', 20)
            )
        elif db_type == 'neo4j':
            # Neo4j graph database service
            from rag_factory.services.database import Neo4jDatabaseService
            return Neo4jDatabaseService(
                uri=config.get('uri', f"bolt://{config['host']}:{config.get('port', 7687)}"),
                user=config['user'],
                password=config['password']
            )
        else:
            raise ServiceInstantiationError(
                f"Unknown database type: {db_type}"
            )
```

### Custom Exceptions
```python
# rag_factory/registry/exceptions.py

class ServiceRegistryError(Exception):
    """Base exception for service registry errors."""
    pass


class ServiceNotFoundError(ServiceRegistryError):
    """Service not found in registry."""
    pass


class ServiceInstantiationError(ServiceRegistryError):
    """Service instantiation failed."""
    pass
```

---

## Unit Tests

### Test File Locations
- `tests/unit/registry/test_service_registry.py`
- `tests/unit/registry/test_service_factory.py`
- `tests/unit/registry/test_threading.py`

### Test Cases

#### TC17.2.1: Service Registry Tests
```python
import pytest
from unittest.mock import Mock, patch, mock_open
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.registry.exceptions import ServiceNotFoundError, ServiceInstantiationError

@pytest.fixture
def services_yaml_content():
    return """
services:
  llm1:
    name: "test-llm"
    url: "http://localhost:1234/v1"
    api_key: "test-key"
    model: "test-model"
    temperature: 0.7

  embedding1:
    name: "test-embedding"
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

def test_registry_initialization(mock_services_file):
    """Test service registry initialization."""
    registry = ServiceRegistry(mock_services_file)

    assert registry.config is not None
    assert 'services' in registry.config
    assert len(registry.list_services()) == 3

def test_list_services(mock_services_file):
    """Test listing available services."""
    registry = ServiceRegistry(mock_services_file)

    services = registry.list_services()

    assert 'llm1' in services
    assert 'embedding1' in services
    assert 'db1' in services
    assert len(services) == 3

def test_get_service_creates_instance(mock_services_file):
    """Test getting service creates instance on first call."""
    registry = ServiceRegistry(mock_services_file)

    with patch.object(registry._factory, 'create_service') as mock_create:
        mock_service = Mock()
        mock_create.return_value = mock_service

        service = registry.get("llm1")

        assert service is mock_service
        mock_create.assert_called_once()

def test_get_service_caches_instance(mock_services_file):
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

def test_get_service_with_dollar_prefix(mock_services_file):
    """Test service lookup with $ prefix."""
    registry = ServiceRegistry(mock_services_file)

    with patch.object(registry._factory, 'create_service') as mock_create:
        mock_service = Mock()
        mock_create.return_value = mock_service

        # Both should work
        service1 = registry.get("$llm1")
        service2 = registry.get("llm1")

        assert service1 is service2

def test_get_nonexistent_service_raises_error(mock_services_file):
    """Test getting non-existent service raises error."""
    registry = ServiceRegistry(mock_services_file)

    with pytest.raises(ServiceNotFoundError) as exc_info:
        registry.get("nonexistent")

    assert "nonexistent" in str(exc_info.value)
    assert "llm1" in str(exc_info.value)  # Should suggest available services

def test_reload_service(mock_services_file):
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

def test_shutdown_closes_all_services(mock_services_file):
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

def test_context_manager(mock_services_file):
    """Test registry works as context manager."""
    with patch('rag_factory.registry.service_registry.ServiceRegistry.shutdown') as mock_shutdown:
        with ServiceRegistry(mock_services_file) as registry:
            assert registry is not None

        mock_shutdown.assert_called_once()
```

#### TC17.2.2: Service Factory Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.registry.service_factory import ServiceFactory
from rag_factory.registry.exceptions import ServiceInstantiationError

@pytest.fixture
def factory():
    return ServiceFactory()

def test_create_llm_service_lm_studio(factory):
    """Test creating LM Studio LLM service."""
    config = {
        "name": "test-llm",
        "url": "http://localhost:1234/v1",
        "model": "test-model",
        "temperature": 0.7
    }

    with patch('rag_factory.registry.service_factory.LMStudioLLMService') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        service = factory.create_service("llm1", config)

        assert service is mock_instance
        mock_class.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            model="test-model",
            temperature=0.7,
            timeout=30
        )

def test_create_llm_service_openai(factory):
    """Test creating OpenAI LLM service."""
    config = {
        "name": "openai-llm",
        "url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-4"
    }

    with patch('rag_factory.registry.service_factory.OpenAILLMService') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        service = factory.create_service("llm_openai", config)

        assert service is mock_instance
        mock_class.assert_called_once()

def test_create_embedding_service_onnx(factory):
    """Test creating ONNX embedding service."""
    config = {
        "name": "onnx-embed",
        "provider": "onnx",
        "model": "Xenova/all-MiniLM-L6-v2",
        "cache_dir": "./models",
        "batch_size": 32
    }

    with patch('rag_factory.registry.service_factory.ONNXEmbeddingService') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        service = factory.create_service("embed1", config)

        assert service is mock_instance
        mock_class.assert_called_once_with(
            model_name="Xenova/all-MiniLM-L6-v2",
            cache_dir="./models",
            batch_size=32
        )

def test_create_database_service_postgres(factory):
    """Test creating PostgreSQL database service."""
    config = {
        "name": "postgres-db",
        "type": "postgres",
        "connection_string": "postgresql://user:pass@localhost:5432/db",
        "pool_size": 10,
        "max_overflow": 20
    }

    with patch('rag_factory.registry.service_factory.PostgresqlDatabaseService') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        service = factory.create_service("db1", config)

        assert service is mock_instance
        mock_class.assert_called_once_with(
            connection_string="postgresql://user:pass@localhost:5432/db",
            pool_size=10,
            max_overflow=20
        )

def test_create_service_unknown_type(factory):
    """Test creating service with unknown type raises error."""
    config = {
        "name": "unknown",
        "unknown_key": "unknown_value"
    }

    with pytest.raises(ServiceInstantiationError) as exc_info:
        factory.create_service("unknown", config)

    assert "Cannot determine service type" in str(exc_info.value)
```

#### TC17.2.3: Thread Safety Tests
```python
import pytest
import threading
import time
from unittest.mock import Mock, patch
from rag_factory.registry.service_registry import ServiceRegistry

def test_concurrent_service_access(mock_services_file):
    """Test concurrent access to same service creates only one instance."""
    registry = ServiceRegistry(mock_services_file)

    # Track number of times create_service is called
    call_count = 0
    call_lock = threading.Lock()

    def slow_create_service(service_name, config):
        """Simulate slow service creation."""
        nonlocal call_count
        time.sleep(0.1)  # Simulate slow creation
        with call_lock:
            call_count += 1
        return Mock()

    with patch.object(registry._factory, 'create_service', side_effect=slow_create_service):
        # Launch multiple threads accessing same service
        threads = []
        results = []

        def get_service():
            service = registry.get("llm1")
            results.append(service)

        for _ in range(10):
            thread = threading.Thread(target=get_service)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should only create service once despite concurrent access
        assert call_count == 1

        # All threads should get same instance
        assert len(set(id(s) for s in results)) == 1

def test_concurrent_different_services(mock_services_file):
    """Test concurrent access to different services works correctly."""
    registry = ServiceRegistry(mock_services_file)

    with patch.object(registry._factory, 'create_service', return_value=Mock()):
        results = {
            'llm1': [],
            'embedding1': [],
            'db1': []
        }

        def get_service(service_name):
            service = registry.get(service_name)
            results[service_name].append(service)

        threads = []
        for service_name in ['llm1', 'embedding1', 'db1']:
            for _ in range(5):
                thread = threading.Thread(target=get_service, args=(service_name,))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        # Each service should have consistent instances
        for service_name, instances in results.items():
            assert len(instances) == 5
            assert len(set(id(i) for i in instances)) == 1
```

---

## Integration Tests

```python
# tests/integration/registry/test_registry_integration.py

@pytest.mark.integration
def test_real_service_instantiation(tmp_path):
    """Test instantiating real services from configuration."""
    services_yaml = tmp_path / "services.yaml"
    services_yaml.write_text("""
services:
  embedding_local:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384
""")

    registry = ServiceRegistry(str(services_yaml))

    # Get embedding service
    embedding = registry.get("embedding_local")

    assert embedding is not None
    assert hasattr(embedding, 'embed')

    # Test caching
    embedding2 = registry.get("embedding_local")
    assert embedding is embedding2

    registry.shutdown()

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="API key not set")
def test_openai_service_instantiation(tmp_path):
    """Test instantiating OpenAI service."""
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    services_yaml = tmp_path / "services.yaml"
    services_yaml.write_text("""
services:
  llm_openai:
    name: "openai-gpt4"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
""")

    registry = ServiceRegistry(str(services_yaml))

    llm = registry.get("llm_openai")

    assert llm is not None
    assert hasattr(llm, 'complete')

    registry.shutdown()
```

---

## Definition of Done

- [ ] ServiceRegistry class implemented
- [ ] ServiceFactory class implemented
- [ ] Service loading from YAML working
- [ ] Service caching working (singleton per definition)
- [ ] Thread-safe instantiation with locks
- [ ] Environment variable resolution integrated
- [ ] Support for LLM services (LM Studio, OpenAI)
- [ ] Support for embedding services (ONNX, OpenAI, Cohere)
- [ ] Support for database services (PostgreSQL)
- [ ] Service lifecycle management (shutdown, reload)
- [ ] Custom exceptions defined
- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass with real services
- [ ] Thread safety tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Uses existing dependencies from Story 17.1
# No new dependencies required
```

### Usage Example

```python
from rag_factory.registry.service_registry import ServiceRegistry

# Initialize registry
registry = ServiceRegistry("config/services.yaml")

# Get services (creates on first call, returns cached on subsequent calls)
llm = registry.get("$llm1")           # Creates LLM service
embedding = registry.get("embedding1")  # Creates embedding service ($ optional)
db = registry.get("db1")               # Creates database service

# Multiple strategies share same instances
strategy1 = Strategy1(llm=llm, embedding=embedding, db=db)
strategy2 = Strategy2(llm=llm, embedding=embedding, db=db)  # Same instances!

# Verify they're actually the same object
assert strategy1.llm is strategy2.llm  # True - same instance

# List available services
print(registry.list_services())  # ['llm1', 'embedding1', 'db1']

# Reload a service (after config change)
new_llm = registry.reload("llm1")

# Cleanup (or use context manager)
registry.shutdown()

# Context manager usage
with ServiceRegistry("config/services.yaml") as registry:
    llm = registry.get("llm1")
    # Use services...
# Automatic cleanup on exit
```

---

## Notes for Developers

1. **Thread Safety**: Per-service locks prevent race conditions without blocking unrelated services.

2. **Lazy Loading**: Services are only created when first requested, reducing startup time.

3. **Memory Management**: Only one instance per service definition prevents memory waste.

4. **Error Handling**: Always provide clear error messages with available service lists.

5. **Service Cleanup**: Always call shutdown() or use context manager to prevent resource leaks.

6. **Configuration Reloading**: Use reload() for development, but be careful in production.

7. **Testing**: Mock ServiceFactory.create_service() to avoid creating real services in unit tests.

8. **Performance**: Service lookup is O(1) from cache, but first instantiation can be slow (model loading).

9. **Environment Variables**: Must be set before registry initialization. Changes require reload.

10. **Integration**: Works seamlessly with existing Epic 11 service interfaces - no code changes needed.
