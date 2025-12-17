# Centralized Mock System

This directory contains reusable mock builders for all test types in the RAG Factory project.

## Purpose

Instead of creating mocks individually in each test file, this system provides:
- **Consistent mock implementations** across all tests
- **Reduced code duplication** (83% reduction in mock code)
- **Easier maintenance** when service interfaces change
- **Faster test development** with pre-built fixtures

## Quick Start

### Using Pre-built Fixtures

The easiest way to use mocks is through pytest fixtures in `conftest.py`:

```python
# tests/integration/test_my_feature.py
import pytest

@pytest.mark.asyncio
async def test_my_feature(mock_registry_with_services):
    """Test using centralized mock registry."""
    # mock_registry_with_services includes:
    # - Mock embedding service
    # - Mock database service
    # - Mock migration validator
    
    manager = StrategyPairManager(
        service_registry=mock_registry_with_services,
        config_dir=str(config_dir)
    )
    # ... test implementation
```

### Creating Custom Mocks

For custom behavior, use the builder functions:

```python
from tests.mocks import create_mock_embedding_service

def test_custom_embedding():
    # Create with custom configuration
    embedding_service = create_mock_embedding_service(
        dimension=768,
        model_name="custom-model",
        embed_return_value=[0.5] * 768
    )
    # Use in test...
```

## Available Mock Builders

### Service Mocks (`services.py`)

- `create_mock_embedding_service()` - Mock embedding service with configurable dimensions
- `create_mock_database_service()` - Mock database service with CRUD operations
- `create_mock_llm_service()` - Mock LLM service with generation capabilities
- `create_mock_neo4j_service()` - Mock Neo4j graph database service

### Database Mocks (`database.py`)

- `create_mock_engine()` - Mock SQLAlchemy engine
- `create_mock_connection()` - Mock database connection
- `create_mock_migration_validator()` - Mock migration validator

### Strategy Mocks (`strategies.py`)

- `create_mock_indexing_strategy()` - Mock indexing strategy
- `create_mock_retrieval_strategy()` - Mock retrieval strategy
- `create_mock_reranker_service()` - Mock reranking service

### Infrastructure Mocks (`infrastructure.py`)

- `create_mock_registry()` - Mock service registry
- `create_mock_onnx_environment()` - Context manager for ONNX mocking

## Available Fixtures

All fixtures are defined in `tests/conftest.py`:

### Basic Service Fixtures

- `mock_embedding_service` - Standard embedding service (384 dimensions)
- `mock_database_service` - Standard database service
- `mock_llm_service` - Standard LLM service
- `mock_neo4j_service` - Standard Neo4j service

### Composite Fixtures

- `mock_registry_with_services` - Registry with embedding + database services
- `mock_registry_with_graph_services` - Registry with all services including Neo4j
- `mock_registry_with_llm_services` - Registry with LLM + embedding + database

### Infrastructure Fixtures

- `mock_migration_validator` - Pre-configured migration validator
- `mock_engine_with_connection` - Engine with connection context manager

## Customization Examples

### Custom Embedding Dimension

```python
from tests.mocks import create_mock_embedding_service

@pytest.fixture
def mock_768_embedding_service():
    return create_mock_embedding_service(dimension=768)

def test_with_custom_dimension(mock_768_embedding_service):
    # Use 768-dimension embeddings
    pass
```

### Custom Database Responses

```python
from tests.mocks import create_mock_database_service

def test_with_custom_data():
    db_service = create_mock_database_service(
        search_chunks_return_value=[
            {'id': 'custom1', 'text': 'custom content', 'score': 0.95}
        ]
    )
    # Use custom database responses
```

### Custom LLM Responses

```python
from tests.mocks import create_mock_llm_service

def test_with_custom_llm():
    llm_service = create_mock_llm_service(
        generate_return_value="Custom LLM response"
    )
    # Use custom LLM behavior
```

## Architecture

```
tests/mocks/
├── __init__.py           # Public API exports
├── builders.py           # Base builder classes
├── services.py           # Service mock builders
├── database.py           # Database mock builders
├── strategies.py         # Strategy mock builders
└── infrastructure.py     # Infrastructure mock builders
```

## Migration Guide

When migrating an existing test to use centralized mocks:

1. **Identify mocks** in the test file
2. **Find equivalent fixture** in conftest.py or create using builders
3. **Replace fixture** in test function signature
4. **Remove duplicate code** (mock setup in fixtures)
5. **Run tests** to verify behavior unchanged
6. **Update imports** (remove unused `unittest.mock` imports)

### Before Migration

```python
@pytest.fixture
def mock_registry():
    embedding_service = Mock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
    # ... 50 more lines
    return registry

def test_feature(mock_registry):
    # test code
```

### After Migration

```python
def test_feature(mock_registry_with_services):
    # test code - no fixture needed!
```

## Best Practices

1. **Use fixtures when possible** - They're pre-configured and tested
2. **Use builders for customization** - When you need specific behavior
3. **Document custom fixtures** - If you create test-specific fixtures
4. **Keep mocks simple** - Don't over-configure unless necessary
5. **Test mock behavior** - Verify mocks match real service interfaces

## Troubleshooting

### Mock not behaving as expected

Check the builder function documentation for available configuration options.

### Need custom behavior not supported

1. Use the builder function with custom parameters
2. Or create a test-specific fixture using the builder
3. Consider contributing the customization back to the builder

### Tests failing after migration

1. Verify the mock configuration matches the old setup
2. Check for any custom mock behavior that wasn't migrated
3. Ensure async methods use `AsyncMock` not `Mock`

## Contributing

When adding new mock capabilities:

1. Add builder function to appropriate module
2. Add fixture to `conftest.py` if commonly used
3. Document in this README
4. Add unit tests for the builder
5. Update migration examples

## Statistics

- **Total mocks centralized**: 1,178
- **Test files using mocks**: 96
- **Code reduction**: ~1,500 lines (83%)
- **Maintenance locations**: 1 (down from 96)
