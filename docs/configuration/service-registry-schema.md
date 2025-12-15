# Service Registry Schema Reference

## Overview

The service registry is a YAML configuration file that defines reusable services for RAG strategies. Services can be LLMs, embedding providers, or databases. This document provides a complete reference for the service registry schema.

## Schema Version

Current version: **1.0.0**

## File Format

```yaml
version: "1.0.0"  # Optional: Schema version
services:
  service_name:
    # Service configuration
```

## Service Types

### LLM Service

LLM services provide language model capabilities for query expansion, reranking, and generation.

#### Required Fields

- `name` (string): Human-readable service name
- `type` (string): Must be `"llm"`

#### Optional Fields

- `url` (string, URI format): LLM API endpoint
- `api_key` (string): API key (use environment variables: `${API_KEY}`)
- `model` (string): Model identifier (e.g., `"gpt-4"`, `"claude-3-opus"`)
- `temperature` (number, 0-2): Sampling temperature (default: 0.7)
- `max_tokens` (integer, ≥1): Maximum tokens in response
- `timeout` (integer, ≥1): Request timeout in seconds (default: 30)

#### Example

```yaml
services:
  openai_gpt4:
    name: "OpenAI GPT-4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    timeout: 60

  local_llm:
    name: "Local LLM Server"
    type: "llm"
    url: "http://localhost:1234/v1"
    model: "mistral-7b-instruct"
    temperature: 0.5
```

### Embedding Service

Embedding services convert text into vector representations for semantic search.

#### Required Fields

- `name` (string): Human-readable service name
- `type` (string): Must be `"embedding"`
- `provider` (string): Embedding provider, one of:
  - `"onnx"` - Local ONNX runtime
  - `"openai"` - OpenAI embeddings API
  - `"cohere"` - Cohere embeddings API
  - `"huggingface"` - HuggingFace models

#### Optional Fields

- `model` (string): Model name or path
- `cache_dir` (string): Directory for model cache (local providers)
- `batch_size` (integer, ≥1): Batch size for embedding (default: 32)
- `dimensions` (integer, ≥1): Embedding vector dimensions
- `api_key` (string): API key for cloud providers (use environment variables)

#### Examples

```yaml
services:
  # Local ONNX embedding
  embedding_local:
    name: "Local ONNX MiniLM"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384

  # OpenAI embedding
  embedding_openai:
    name: "OpenAI text-embedding-3-small"
    type: "embedding"
    provider: "openai"
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimensions: 1536

  # Cohere embedding
  embedding_cohere:
    name: "Cohere Embed v3"
    type: "embedding"
    provider: "cohere"
    model: "embed-english-v3.0"
    api_key: "${COHERE_API_KEY}"
    dimensions: 1024
```

### Database Service

Database services provide storage for documents, chunks, and vector embeddings.

#### Required Fields

- `name` (string): Human-readable service name
- `type` (string): Database type, one of:
  - `"database"` - Generic database
  - `"postgres"` - PostgreSQL with pgvector
  - `"neo4j"` - Neo4j graph database
  - `"mongodb"` - MongoDB

#### Optional Fields

- `connection_string` (string): Full database connection string
- `host` (string): Database host
- `port` (integer): Database port
- `database` (string): Database name
- `user` (string): Database user
- `password` (string): Database password (use environment variables)
- `pool_size` (integer, ≥1): Connection pool size (default: 10)
- `max_overflow` (integer, ≥0): Maximum overflow connections (default: 20)

#### Examples

```yaml
services:
  # PostgreSQL with connection string
  db_main:
    name: "Main PostgreSQL Database"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20

  # PostgreSQL with individual parameters
  db_postgres:
    name: "PostgreSQL Database"
    type: "postgres"
    host: "${DB_HOST:-localhost}"
    port: 5432
    database: "rag_factory"
    user: "${DB_USER}"
    password: "${DB_PASSWORD}"
    pool_size: 15

  # Neo4j graph database
  db_neo4j:
    name: "Neo4j Knowledge Graph"
    type: "neo4j"
    connection_string: "${NEO4J_URL}"
    user: "${NEO4J_USER}"
    password: "${NEO4J_PASSWORD}"
```

## Environment Variables

All configuration values support environment variable substitution using the following syntax:

### Syntax

1. **Required variable**: `${VAR_NAME}`
   - Raises error if variable is not set
   - Example: `${DATABASE_URL}`

2. **Optional with default**: `${VAR_NAME:-default_value}`
   - Uses default value if variable is not set
   - Example: `${DB_HOST:-localhost}`

3. **Required with custom error**: `${VAR_NAME:?error message}`
   - Raises error with custom message if variable is not set
   - Example: `${API_KEY:?OpenAI API key is required}`

### Examples

```yaml
services:
  example_service:
    name: "Example Service"
    type: "llm"
    # Required variable
    api_key: "${OPENAI_API_KEY}"
    
    # Optional with default
    url: "${LLM_URL:-http://localhost:1234/v1}"
    
    # Required with custom error
    model: "${MODEL_NAME:?Model name must be specified}"
    
    # Partial string replacement
    connection_string: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
```

## Validation

### Schema Validation

The configuration is validated against a JSON Schema. Common validation errors:

- **Missing required field**: Ensure all required fields are present
- **Invalid field type**: Check that field values match expected types
- **Invalid enum value**: Verify enum values (e.g., `provider`, `type`)
- **Invalid format**: Check URI format for URLs

### Security Warnings

The validator checks for potential security issues:

- **Plaintext secrets**: Warns if API keys or passwords appear to be plaintext
  - Always use environment variables: `${API_KEY}`
  - Never commit secrets to version control

### Example Validation

```python
from rag_factory.config.validator import load_yaml_with_validation
from rag_factory.config.env_resolver import EnvResolver

# Load and validate
config = load_yaml_with_validation(
    "config/services.yaml",
    config_type="service_registry"
)

# Resolve environment variables
config = EnvResolver.resolve(config)
```

## Complete Example

```yaml
version: "1.0.0"

services:
  # Local embedding service
  embedding_local:
    name: "Local ONNX MiniLM"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384

  # Cloud embedding service
  embedding_openai:
    name: "OpenAI Embeddings"
    type: "embedding"
    provider: "openai"
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimensions: 1536

  # Local LLM
  llm_local:
    name: "Local Mistral"
    type: "llm"
    url: "${LLM_URL:-http://localhost:1234/v1}"
    model: "mistral-7b-instruct"
    temperature: 0.7
    max_tokens: 2000

  # Cloud LLM
  llm_openai:
    name: "OpenAI GPT-4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    timeout: 60

  # PostgreSQL database
  db_main:
    name: "Main Database"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20

  # Neo4j graph database
  db_neo4j:
    name: "Knowledge Graph"
    type: "neo4j"
    connection_string: "${NEO4J_URL}"
    user: "${NEO4J_USER}"
    password: "${NEO4J_PASSWORD}"
```

## Best Practices

1. **Use descriptive service names**: Choose names that clearly indicate the service purpose
2. **Always use environment variables for secrets**: Never hardcode API keys or passwords
3. **Provide defaults for optional settings**: Use `${VAR:-default}` for better developer experience
4. **Document required environment variables**: Maintain a `.env.example` file
5. **Version your configuration**: Include version field for backward compatibility
6. **Validate before deployment**: Run validation on all configuration files
7. **Use consistent naming**: Follow naming conventions (e.g., `embedding_*`, `llm_*`, `db_*`)

## Troubleshooting

### Common Errors

1. **"Schema validation failed: 'name' is a required property"**
   - Solution: Add the `name` field to the service configuration

2. **"Schema validation failed: 'xyz' is not one of ['onnx', 'openai', 'cohere', 'huggingface']"**
   - Solution: Use a valid provider value

3. **"WARNING: Potential plaintext secret in services.llm1.api_key"**
   - Solution: Replace plaintext value with environment variable: `${API_KEY}`

4. **"Required environment variable ${DATABASE_URL} is not set"**
   - Solution: Set the environment variable or provide a default value

### Getting Help

- See `environment-variables.md` for environment variable guide
- See `strategy-pair-schema.md` for strategy pair configuration
- See example files in `rag_factory/config/examples/`
