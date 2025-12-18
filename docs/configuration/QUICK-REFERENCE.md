# Configuration Quick Reference

## Overview

RAG Factory uses YAML-based configuration with JSON Schema validation and environment variable support.

## File Structure

```
config/
├── services.yaml              # Service registry (LLM, embedding, database)
└── strategies/
    ├── semantic-local.yaml    # Strategy pair configurations
    ├── hybrid-search.yaml
    └── knowledge-graph.yaml
```

## Service Registry (`services.yaml`)

### LLM Service

```yaml
services:
  llm_openai:
    name: "OpenAI GPT-4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    timeout: 60
```

**Required**: `name`, `type`  
**Optional**: `url`, `api_key`, `model`, `temperature`, `max_tokens`, `timeout`

### Embedding Service

```yaml
services:
  embedding_local:
    name: "Local ONNX MiniLM"
    type: "embedding"
    provider: "onnx"                    # onnx, openai, cohere, huggingface
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384
```

**Required**: `name`, `type`, `provider`  
**Optional**: `model`, `cache_dir`, `batch_size`, `dimensions`, `api_key`

### Database Service

```yaml
services:
  db_main:
    name: "Main Database"
    type: "postgres"                    # postgres, neo4j, mongodb
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
```

**Required**: `name`, `type`  
**Optional**: `connection_string`, `host`, `port`, `database`, `user`, `password`, `pool_size`, `max_overflow`

## Strategy Pair Configuration

### Basic Structure

```yaml
strategy_name: "semantic-local"         # lowercase, numbers, hyphens only
version: "1.0.0"                        # semantic versioning
description: "Basic semantic search"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"       # Service reference
    database: "$db_main"
  db_config:
    tables:
      documents: "documents"
      chunks: "chunks"
  config:
    chunk_size: 512
    chunk_overlap: 50

retriever:
  strategy: "SemanticRetriever"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  config:
    top_k: 5
    similarity_threshold: 0.7

migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"

expected_schema:
  tables: ["documents", "chunks"]
  indexes: ["idx_chunks_embedding"]
  extensions: ["vector"]

tags: ["semantic-search", "local", "basic"]
```

## Environment Variables

### Syntax

```yaml
# Required variable (error if not set)
api_key: "${OPENAI_API_KEY}"

# Optional with default
host: "${DB_HOST:-localhost}"

# Required with custom error
api_key: "${API_KEY:?API key is required}"

# Partial string replacement
connection: "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
```

### Setup (.env file)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_factory
DB_HOST=localhost
DB_PORT=5432

# API Keys
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...

# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## Loading Configuration

### Python API

```python
from rag_factory.config import load_yaml_with_validation, EnvResolver
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Load service registry
services = load_yaml_with_validation(
    "config/services.yaml",
    config_type="service_registry"
)
services = EnvResolver.resolve(services)

# Load strategy pair
with open("config/services.yaml") as f:
    service_registry = yaml.safe_load(f)

strategy = load_yaml_with_validation(
    "config/strategies/semantic-local.yaml",
    config_type="strategy_pair",
    service_registry=service_registry
)
strategy = EnvResolver.resolve(strategy)
```

### Validation Only

```python
from rag_factory.config.validator import ConfigValidator

validator = ConfigValidator()

# Validate service registry
warnings = validator.validate_services_yaml(config, "config/services.yaml")

# Validate strategy pair
warnings = validator.validate_strategy_pair_yaml(
    strategy_config,
    service_registry=services,
    file_path="config/strategy.yaml"
)
```

## Service Reference

### Using References

```yaml
# Define service once in services.yaml
services:
  embedding_local:
    name: "Local Embedding"
    type: "embedding"
    provider: "onnx"

# Reference in strategy pair
indexer:
  services:
    embedding: "$embedding_local"  # $ prefix required
```

### Inline Configuration

```yaml
indexer:
  services:
    embedding:
      name: "Inline Embedding"
      type: "embedding"
      provider: "onnx"
      model: "all-MiniLM-L6-v2"
```

## Database Mapping

### Table Mapping

```yaml
db_config:
  tables:
    documents: "my_documents"      # logical: physical
    chunks: "my_chunks"
    metadata: "my_metadata"
```

### Field Mapping

```yaml
db_config:
  fields:
    content: "text_content"        # logical: physical
    embedding: "vector_embedding"
    document_id: "doc_id"
```

## Common Patterns

### Local Development

```yaml
services:
  embedding_local:
    name: "Local ONNX"
    type: "embedding"
    provider: "onnx"
    model: "${MODEL_NAME:-Xenova/all-MiniLM-L6-v2}"
    cache_dir: "${CACHE_DIR:-./models}"
  
  db_dev:
    name: "Dev Database"
    type: "postgres"
    host: "${DB_HOST:-localhost}"
    port: "${DB_PORT:-5432}"
    database: "${DB_NAME:-rag_factory_dev}"
    user: "${DB_USER:-postgres}"
    password: "${DB_PASSWORD}"
```

### Production

```yaml
services:
  embedding_prod:
    name: "OpenAI Embeddings"
    type: "embedding"
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"
  
  db_prod:
    name: "Production Database"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 20
    max_overflow: 40
```

### Hybrid Search

```yaml
strategy_name: "hybrid-search"
version: "1.0.0"

indexer:
  strategy: "HybridIndexer"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  config:
    chunk_size: 512
    extract_keywords: true

retriever:
  strategy: "HybridRetriever"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  config:
    top_k: 10
    semantic_weight: 0.7
    keyword_weight: 0.3
```

## Validation

### Command Line

```bash
# Validate service registry
python -c "
from rag_factory.config.validator import load_yaml_with_validation
load_yaml_with_validation('config/services.yaml', 'service_registry')
print('✓ Valid')
"

# Check required environment variables
python -c "
from rag_factory.config.env_resolver import EnvResolver
import yaml
with open('config/services.yaml') as f:
    config = yaml.safe_load(f)
print('Required:', EnvResolver.extract_variable_names(config))
"
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `'name' is a required property` | Missing required field | Add the field |
| `'xyz' is not one of [...]` | Invalid enum value | Use valid value |
| `Service reference '$xyz' not found` | Service doesn't exist | Add to services.yaml |
| `${VAR} is not set` | Missing environment variable | Set the variable |
| `does not match '^[a-z0-9-]+$'` | Invalid strategy name | Use lowercase + hyphens |

## Best Practices

### Security
- ✅ Always use environment variables for secrets
- ✅ Never commit `.env` files
- ✅ Provide `.env.example` template
- ✅ Use secret management in production

### Configuration
- ✅ Define services once, reference everywhere
- ✅ Use descriptive service names
- ✅ Provide defaults for optional settings
- ✅ Version your configurations
- ✅ Document required environment variables

### Validation
- ✅ Validate before deployment
- ✅ Test with missing environment variables
- ✅ Check service references
- ✅ Verify database mappings

## Examples

See complete examples in:
- `rag_factory/config/examples/services.yaml`
- `rag_factory/config/examples/semantic-local-pair.yaml`
- `rag_factory/config/examples/hybrid-search-pair.yaml`

## Documentation

Detailed guides:
- [Service Registry Schema](service-registry-schema.md)
- [Strategy Pair Schema](strategy-pair-schema.md)
- [Environment Variables](environment-variables.md)
- [Troubleshooting](troubleshooting.md)

## Quick Start

1. **Copy example configuration**:
   ```bash
   cp rag_factory/config/examples/services.yaml config/services.yaml
   ```

2. **Create .env file**:
   ```bash
   cat > .env << EOF
   DATABASE_URL=postgresql://localhost/rag_factory
   OPENAI_API_KEY=sk-your-key
   EOF
   ```

3. **Load and use**:

```python
from dotenv import load_dotenv
from rag_factory.config import load_yaml_with_validation, EnvResolver


load_dotenv()
config = load_yaml_with_validation("config/services.yaml", "service_registry")
config = EnvResolver.resolve(config)
   ```

## Support

- Examples: `rag_factory/config/examples/`
- Tests: `tests/unit/config/`, `tests/integration/config/`
- Documentation: `docs/configuration/`
