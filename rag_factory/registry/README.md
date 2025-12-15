# Service Registry

The Service Registry is a central component of the RAG Factory that manages service instances and enables efficient resource sharing across multiple strategies.

## Overview

The Service Registry provides:
- **Lazy Instantiation**: Services are only created when first requested
- **Singleton Pattern**: One instance per service definition (shared across strategies)
- **Thread Safety**: Concurrent access is handled safely with per-service locks
- **Environment Variable Resolution**: Automatic resolution of `${VAR}` syntax in configurations
- **Lifecycle Management**: Proper cleanup and resource management

## Quick Start

### Basic Usage

```python
from rag_factory.registry import ServiceRegistry

# Initialize registry from configuration file
registry = ServiceRegistry("config/services.yaml")

# Get services (creates on first call, returns cached on subsequent calls)
llm = registry.get("llm_openai")
embedding = registry.get("embedding_local")
database = registry.get("db_postgres")

# Multiple strategies can share the same instances
strategy1 = MyStrategy(llm=llm, embedding=embedding)
strategy2 = AnotherStrategy(llm=llm, embedding=embedding)  # Same instances!

# Cleanup
registry.shutdown()
```

### Context Manager Usage

```python
with ServiceRegistry("config/services.yaml") as registry:
    llm = registry.get("llm_openai")
    embedding = registry.get("embedding_local")
    
    # Use services...
    
# Automatic cleanup on exit
```

## Configuration

Services are defined in `services.yaml`:

```yaml
services:
  llm_openai:
    name: "openai-gpt4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"

  embedding_local:
    name: "local-onnx-minilm"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32

  db_postgres:
    name: "main-database"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
```

### Environment Variables

The registry supports three environment variable patterns:

- `${VAR}` - Required variable (error if not set)
- `${VAR:-default}` - Optional with default value
- `${VAR:?error message}` - Required with custom error message

## Supported Services

### LLM Services

Currently supported:
- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.

Configuration:
```yaml
llm_openai:
  name: "openai-gpt4"
  type: "llm"
  url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
```

**Note**: LM Studio and other OpenAI-compatible services are not yet fully supported.

### Embedding Services

Supported providers:
- **ONNX**: Local embedding models (e.g., MiniLM, BGE)
- **OpenAI**: text-embedding-3-small, text-embedding-ada-002

Configuration:
```yaml
# ONNX (local)
embedding_local:
  name: "local-onnx"
  type: "embedding"
  provider: "onnx"
  model: "Xenova/all-MiniLM-L6-v2"
  cache_dir: "./models"
  batch_size: 32

# OpenAI
embedding_openai:
  name: "openai-embeddings"
  type: "embedding"
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "text-embedding-3-small"
```

### Database Services

Supported databases:
- **PostgreSQL**: With pgvector extension
- **Neo4j**: Graph database for knowledge graphs

Configuration:
```yaml
# PostgreSQL
db_postgres:
  name: "main-db"
  type: "postgres"
  connection_string: "postgresql://user:pass@localhost:5432/db"
  pool_size: 10
  max_overflow: 20

# Neo4j
db_neo4j:
  name: "knowledge-graph"
  type: "neo4j"
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"
```

## API Reference

### ServiceRegistry

#### `__init__(config_path: str)`
Initialize the service registry from a YAML configuration file.

**Parameters:**
- `config_path`: Path to services.yaml file

**Raises:**
- `ServiceInstantiationError`: If configuration file not found or invalid

#### `get(service_ref: str) -> Any`
Get or create a service instance.

**Parameters:**
- `service_ref`: Service name (with or without `$` prefix)

**Returns:**
- Service instance implementing the appropriate interface

**Raises:**
- `ServiceNotFoundError`: If service not found in registry
- `ServiceInstantiationError`: If service creation fails

#### `list_services() -> List[str]`
List all available service names.

**Returns:**
- List of service names from configuration

#### `reload(service_name: str) -> Any`
Force reload a service (useful after config changes).

**Parameters:**
- `service_name`: Service to reload

**Returns:**
- New service instance

#### `shutdown() -> None`
Close all service instances and cleanup resources.

## Thread Safety

The Service Registry is fully thread-safe:

- **Per-Service Locks**: Each service has its own lock, allowing concurrent instantiation of different services
- **Double-Check Locking**: Prevents duplicate instantiation in race conditions
- **No Global Locks**: Different services can be instantiated concurrently

Example:
```python
import threading

registry = ServiceRegistry("config/services.yaml")

def worker(service_name):
    service = registry.get(service_name)
    # Use service...

# Multiple threads can safely access different services
threads = [
    threading.Thread(target=worker, args=("llm_openai",)),
    threading.Thread(target=worker, args=("embedding_local",)),
    threading.Thread(target=worker, args=("db_postgres",))
]

for t in threads:
    t.start()
for t in threads:
    t.join()

registry.shutdown()
```

## Performance

- **Service Lookup**: <10ms (cached)
- **First Instantiation**: Varies by service type
  - ONNX models: 1-5s (model loading)
  - API services: <100ms
  - Database connections: <500ms
- **Memory Overhead**: <50MB for registry itself

## Error Handling

The registry provides clear error messages:

```python
try:
    service = registry.get("nonexistent_service")
except ServiceNotFoundError as e:
    print(e)
    # Output: Service 'nonexistent_service' not found in registry.
    #         Available services: ['llm_openai', 'embedding_local', ...]
```

## Best Practices

1. **Use Context Manager**: Always use `with` statement or call `shutdown()` explicitly
2. **Environment Variables**: Never hardcode secrets in YAML files
3. **Service Naming**: Use descriptive names (e.g., `llm_openai_gpt4` not `llm1`)
4. **Resource Limits**: Configure appropriate pool sizes for database services
5. **Development vs Production**: Use different configurations for different environments

## Examples

See `rag_factory/config/examples/services.yaml` for a comprehensive example configuration.

## Testing

The Service Registry has comprehensive test coverage:

- **Unit Tests**: `tests/unit/registry/`
  - Service factory tests
  - Registry tests
  - Thread safety tests
  - Exception tests

- **Integration Tests**: `tests/integration/registry/`
  - Real service instantiation
  - Environment variable resolution
  - Service lifecycle management

Run tests:
```bash
# Unit tests
pytest tests/unit/registry/ -v

# Integration tests
pytest tests/integration/registry/ -v
```

## Troubleshooting

### Service Not Found
**Error**: `ServiceNotFoundError: Service 'xyz' not found`

**Solution**: Check that the service is defined in `services.yaml` and the name matches exactly.

### Environment Variable Not Set
**Error**: `EnvironmentVariableError: Required environment variable ${API_KEY} is not set`

**Solution**: Set the environment variable or use default syntax: `${API_KEY:-default_value}`

### Instantiation Failure
**Error**: `ServiceInstantiationError: Service instantiation failed`

**Solution**: Check the service configuration and ensure all required fields are present and valid.

## Future Enhancements

Planned features:
- Support for LM Studio and OpenAI-compatible services
- Service health checking
- Automatic service restart on failure
- Service metrics and monitoring
- Hot-reloading of configurations
- Service dependency resolution
