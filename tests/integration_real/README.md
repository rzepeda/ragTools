# Real Integration Tests

This directory contains integration tests that use **real services** (PostgreSQL, Neo4j, LM Studio, embeddings) configured via `.env` variables. Tests automatically skip when services are unavailable.

## Overview

Unlike the tests in `tests/integration/` which use mocked services, these tests verify actual service implementations and integration behavior.

## Test Files

- **`test_database_real.py`** - PostgreSQL database operations, vector search, connection pooling
- **`test_embedding_real.py`** - ONNX, OpenAI, and Cohere embedding generation
- **`test_llm_real.py`** - LLM text generation with LM Studio or OpenAI
- **`test_neo4j_real.py`** - Graph database operations and queries
- **`test_end_to_end_real.py`** - Complete RAG pipelines (indexing → retrieval → generation)

## Requirements

### Required Services

1. **PostgreSQL with pgvector** (for database tests)
   - Set `DB_TEST_DATABASE_URL` in `.env`
   
2. **Embedding Model** (for embedding tests)
   - Set `EMBEDDING_MODEL_NAME` in `.env` (e.g., `Xenova/all-MiniLM-L6-v2`)

3. **LLM Service** (for LLM tests)
   - Either LM Studio: Set `LM_STUDIO_BASE_URL` and `LM_STUDIO_MODEL`
   - Or OpenAI API: Set `OPENAI_API_KEY`

4. **Neo4j** (for graph database tests)
   - Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

### Optional Services

- **OpenAI Embeddings**: Set `OPENAI_API_KEY` (must start with `sk-`)
- **Cohere Embeddings**: Set `COHERE_API_KEY`

## Running Tests

### Run All Real Integration Tests

```bash
./run_multiple_tests_with_env.sh tests/integration_real/ -v -m real_integration
```

### Run Specific Service Tests

```bash
# Only PostgreSQL tests
./run_multiple_tests_with_env.sh tests/integration_real/ -m requires_postgres

# Only embedding tests
./run_multiple_tests_with_env.sh tests/integration_real/ -m requires_embeddings

# Only LLM tests
./run_multiple_tests_with_env.sh tests/integration_real/ -m requires_llm

# Only Neo4j tests
./run_multiple_tests_with_env.sh tests/integration_real/ -m requires_neo4j

# Only tests that don't need API keys
./run_multiple_tests_with_env.sh tests/integration_real/ -m "real_integration and not requires_openai and not requires_cohere"
```

### Run Specific Test Files

```bash
# Database tests only
./run_multiple_tests_with_env.sh tests/integration_real/test_database_real.py

# End-to-end tests only
./run_multiple_tests_with_env.sh tests/integration_real/test_end_to_end_real.py
```

## Auto-Skip Behavior

Tests automatically skip when required services are unavailable:

```
SKIPPED [1] tests/integration_real/conftest.py:115: PostgreSQL not available (DB_TEST_DATABASE_URL not set or unreachable)
SKIPPED [1] tests/integration_real/conftest.py:123: Neo4j not available (NEO4J_URI/USER/PASSWORD not set or unreachable)
SKIPPED [1] tests/integration_real/conftest.py:131: LLM service not available (LM_STUDIO_BASE_URL or OPENAI_API_KEY not set or unreachable)
```

## Pytest Markers

All tests are marked with:
- `@pytest.mark.real_integration` - Indicates real service usage
- `@pytest.mark.requires_postgres` - Requires PostgreSQL
- `@pytest.mark.requires_neo4j` - Requires Neo4j
- `@pytest.mark.requires_llm` - Requires LLM service
- `@pytest.mark.requires_embeddings` - Requires embedding service
- `@pytest.mark.requires_openai` - Requires OpenAI API key
- `@pytest.mark.requires_cohere` - Requires Cohere API key
- `@pytest.mark.slow` - Slow tests (large documents, etc.)

## Configuration

Tests use environment variables from `.env`:

```bash
# PostgreSQL
DB_TEST_DATABASE_URL=postgresql://user:password@host:5432/test_db

# Neo4j
NEO4J_URI=bolt://host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LM Studio
LM_STUDIO_BASE_URL=http://host:1234/v1
LM_STUDIO_MODEL=model-name

# Embeddings
EMBEDDING_MODEL_NAME=Xenova/all-MiniLM-L6-v2
EMBEDDING_MODEL_PATH=models/embeddings

# Optional: OpenAI
OPENAI_API_KEY=sk-...

# Optional: Cohere
COHERE_API_KEY=...
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
jobs:
  real-integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Run real integration tests
        env:
          DB_TEST_DATABASE_URL: postgresql://postgres:test@localhost:5432/test
          EMBEDDING_MODEL_NAME: Xenova/all-MiniLM-L6-v2
        run: |
          ./run_multiple_tests_with_env.sh tests/integration_real/ -m "requires_postgres or requires_embeddings"
```

## Benefits

### Comprehensive Coverage
- ✅ Unit tests (fast, mocked) - `tests/unit/`
- ✅ Integration tests (fast, mocked services) - `tests/integration/`
- ✅ Real integration tests (slow, real services) - `tests/integration_real/`

### Flexible Execution
- Run all tests locally with full setup
- Run subset in CI without external dependencies
- Auto-skip unavailable services

### Production Validation
- Test actual database operations
- Verify real API integrations
- Catch service-specific issues
- Performance benchmarks

## What's Tested

### ✅ Real Service Operations
- Actual PostgreSQL vector similarity search
- Real embedding generation (ONNX, OpenAI, Cohere)
- LLM text generation and streaming
- Neo4j graph database queries
- Complete RAG pipelines

### ❌ What's NOT Tested (Use Mock Tests)
- Configuration validation
- Strategy registration
- Dependency injection wiring
- Basic API contracts
