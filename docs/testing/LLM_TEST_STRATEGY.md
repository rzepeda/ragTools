# LLM Testing Strategy

## Overview

This document explains how to run and configure LLM-related tests in the RAG Factory project.

## Test Categories

### 1. Functional Tests (`@pytest.mark.integration`)
- Test core functionality and correctness
- Use LM Studio from `.env` configuration
- Run in CI/CD
- **Required**: LM Studio running locally

### 2. Benchmarking Tests (`@pytest.mark.benchmark` + `@pytest.mark.requires_cloud_api`)
- Compare performance across providers
- Require cloud API keys
- Optional, for performance analysis
- **Required**: Cloud API keys (OpenAI or Anthropic)

## Environment Variables

### Required for Functional Tests

```bash
# LM Studio Configuration (OpenAI-compatible API)
OPENAI_API_BASE=http://192.168.56.1:1234/v1  # LM Studio endpoint
OPENAI_API_KEY=lm-studio                      # Any value works
OPENAI_MODEL=your-model-name                  # Model loaded in LM Studio
RUN_LLM_TESTS=true
```

### Required for Benchmarking Tests

```bash
# Option 1: OpenAI Cloud API
OPENAI_CLOUD_API_KEY=sk-...
OPENAI_CLOUD_MODEL=gpt-3.5-turbo  # Optional, defaults to gpt-3.5-turbo

# Option 2: Anthropic API
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-haiku-20240307  # Optional
```

> **Note**: Use `OPENAI_CLOUD_API_KEY` for cloud OpenAI to distinguish from LM Studio's `OPENAI_API_KEY`.

## Running Tests

### Run all integration tests (with LM Studio)
```bash
pytest tests/integration/strategies/ -m integration
```

### Run only benchmarking tests
```bash
pytest tests/integration/strategies/ -m benchmark
```

### Run all tests except benchmarking
```bash
pytest tests/integration/strategies/ -m "not benchmark"
```

### Run specific test file
```bash
pytest tests/integration/strategies/test_query_expansion_integration.py
```

### Skip tests requiring cloud APIs
```bash
pytest tests/integration/strategies/ -m "not requires_cloud_api"
```

## Available Fixtures

### `llm_service_from_env`
Uses LM Studio configuration from `.env`. Use for functional tests.

**Example:**
```python
@pytest.mark.integration
def test_query_expansion(llm_service_from_env):
    service = QueryExpanderService(config, llm_service_from_env)
    result = service.expand("machine learning")
    assert result.original_query == "machine learning"
```

### `mock_llm_service`
Fully mocked LLM service. Use for fast unit tests that don't need real LLM inference.

**Example:**
```python
def test_with_mock(mock_llm_service):
    # Fast test without real API calls
    response = mock_llm_service.complete([...])
    assert response.text == "Mocked LLM response"
```

### `cloud_llm_service`
Uses cloud providers (OpenAI/Anthropic). Use for benchmarking tests.

**Example:**
```python
@pytest.mark.benchmark
@pytest.mark.requires_cloud_api
def test_performance(cloud_llm_service):
    # Benchmarking test using cloud provider
    pass
```

## Pytest Markers

### `@pytest.mark.integration`
Marks test as an integration test. These tests may use real services (LM Studio, databases, etc.).

### `@pytest.mark.benchmark`
Marks test as a benchmarking test. These are optional performance comparison tests.

### `@pytest.mark.requires_cloud_api`
Marks test as requiring cloud API keys (OpenAI or Anthropic). Tests will be skipped if keys are not available.

### `@pytest.mark.requires_llm`
Marks test as requiring any LLM service (local or cloud).

### `@pytest.mark.database`
Marks test as requiring database connection.

## Example Test Structure

```python
"""Integration tests for my strategy."""

import pytest
from rag_factory.strategies.my_strategy import MyStrategy

@pytest.mark.integration
class TestMyStrategyIntegration:
    """Functional tests using LM Studio."""
    
    def test_basic_functionality(self, llm_service_from_env):
        """Test basic functionality with LM Studio."""
        strategy = MyStrategy(llm_service=llm_service_from_env)
        result = strategy.process("test input")
        assert result is not None
    
    def test_with_mock(self, mock_llm_service):
        """Fast test with mocked LLM."""
        strategy = MyStrategy(llm_service=mock_llm_service)
        result = strategy.process("test input")
        assert result is not None


@pytest.mark.benchmark
@pytest.mark.requires_cloud_api
class TestMyStrategyBenchmark:
    """Benchmarking tests comparing cloud providers."""
    
    def test_performance_comparison(self, cloud_llm_service):
        """Compare performance with cloud providers."""
        strategy = MyStrategy(llm_service=cloud_llm_service)
        # Performance testing logic
        pass
```

## CI/CD Configuration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: ankane/pgvector
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
      
      lm-studio:
        # Use a local LLM service or mock
        image: your-llm-service:latest
        ports:
          - 1234:1234
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests (excluding benchmarks)
        env:
          OPENAI_API_BASE: http://localhost:1234/v1
          OPENAI_API_KEY: lm-studio
          OPENAI_MODEL: test-model
          DB_TEST_DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
        run: |
          pytest tests/ -m "not benchmark and not requires_cloud_api" -v
```

## Troubleshooting

### Tests are skipped with "Cloud API keys not available"

This is expected behavior for benchmarking tests. These tests are optional and only run when you have cloud API keys configured.

**Solution**: Either:
1. Set `OPENAI_CLOUD_API_KEY` or `ANTHROPIC_API_KEY` to run benchmarking tests
2. Or ignore these skipped tests - they're not required for normal development

### Tests fail with "Connection refused" to LM Studio

**Problem**: LM Studio is not running or not accessible.

**Solution**:
1. Start LM Studio and load a model
2. Verify the endpoint: `curl http://192.168.56.1:1234/v1/models`
3. Check `OPENAI_API_BASE` in your `.env` file matches LM Studio's address

### Tests are slow

**Problem**: Using real LLM service for all tests.

**Solution**: Use `mock_llm_service` fixture for unit tests that don't need real LLM inference:

```python
# Slow - uses real LLM
def test_slow(llm_service_from_env):
    pass

# Fast - uses mock
def test_fast(mock_llm_service):
    pass
```

## Best Practices

1. **Use mocks for unit tests**: Reserve real LLM services for integration tests
2. **Mark tests appropriately**: Use `@pytest.mark.integration` and `@pytest.mark.benchmark`
3. **Don't hardcode providers**: Use fixtures instead of creating LLM services directly
4. **Keep benchmarks separate**: Benchmarking tests should be in separate test classes
5. **Document requirements**: Add docstrings explaining what services each test needs

## Migration Guide

### Old Pattern (Hardcoded Provider)

```python
import os
import pytest
from rag_factory.services.llm import LLMService, LLMServiceConfig

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="No API key"
)

@pytest.fixture
def llm_service():
    config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    return LLMService(config)

def test_something(llm_service):
    pass
```

### New Pattern (Using Fixtures)

```python
import pytest

@pytest.mark.integration
def test_something(llm_service_from_env):
    # Uses LM Studio from .env automatically
    pass
```

## Summary

- **Functional tests**: Use `llm_service_from_env` (LM Studio)
- **Unit tests**: Use `mock_llm_service` (fast, no API calls)
- **Benchmarking tests**: Use `cloud_llm_service` (requires cloud API keys)
- **Mark tests**: `@pytest.mark.integration`, `@pytest.mark.benchmark`, `@pytest.mark.requires_cloud_api`
- **CI/CD**: Run tests with `-m "not benchmark and not requires_cloud_api"`
