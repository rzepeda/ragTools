# Configuration Examples

This directory contains example configuration files for the RAG Factory configuration system.

## Files

### services.yaml
Complete service registry example showing:
- Local ONNX embedding service
- Cloud embedding services (OpenAI, Cohere)
- LLM services (local, OpenAI, Anthropic)
- Database services (PostgreSQL, Neo4j)
- Environment variable usage patterns

### semantic-local-pair.yaml
Basic semantic search strategy pair using:
- Local ONNX embeddings
- PostgreSQL with pgvector
- Simple chunking and retrieval

### hybrid-search-pair.yaml
Advanced hybrid search strategy pair combining:
- Semantic vector search
- BM25 keyword search
- Weighted result fusion

## Usage

### 1. Set up environment variables

Create a `.env` file with required variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory

# OpenAI (if using OpenAI services)
OPENAI_API_KEY=sk-...

# Cohere (if using Cohere services)
COHERE_API_KEY=...

# Anthropic (if using Anthropic services)
ANTHROPIC_API_KEY=sk-ant-...

# Neo4j (if using knowledge graph strategies)
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

### 2. Load and validate configurations

```python
from rag_factory.config import load_yaml_with_validation, EnvResolver

# Load service registry
services = load_yaml_with_validation(
    "rag_factory/config/examples/services.yaml",
    config_type="service_registry"
)

# Resolve environment variables
services = EnvResolver.resolve(services)

# Load strategy pair
strategy = load_yaml_with_validation(
    "rag_factory/config/examples/semantic-local-pair.yaml",
    config_type="strategy_pair",
    service_registry=services
)

strategy = EnvResolver.resolve(strategy)
```

### 3. Use in your application

```python
# Access service configurations
embedding_config = services["services"]["embedding_local"]
db_config = services["services"]["db_main"]

# Access strategy configurations
indexer_config = strategy["indexer"]
retriever_config = strategy["retriever"]

# Get database table mappings
tables = indexer_config["db_config"]["tables"]
# {'documents': 'documents', 'chunks': 'chunks'}
```

## Environment Variable Syntax

The configuration system supports three environment variable patterns:

1. **Required variable**: `${VAR_NAME}`
   - Raises error if not set
   - Example: `${DATABASE_URL}`

2. **Optional with default**: `${VAR_NAME:-default_value}`
   - Uses default if not set
   - Example: `${DB_HOST:-localhost}`

3. **Required with custom error**: `${VAR_NAME:?Custom error message}`
   - Raises error with custom message if not set
   - Example: `${API_KEY:?API key is required for this service}`

## Best Practices

1. **Always use environment variables for secrets**
   - Never commit API keys or passwords
   - Use `${ENV_VAR}` syntax in YAML files

2. **Provide sensible defaults**
   - Use `${VAR:-default}` for optional configuration
   - Makes local development easier

3. **Document required variables**
   - List all required environment variables in README
   - Provide `.env.example` file

4. **Validate before deployment**
   - Run validation on all configuration files
   - Check that all required environment variables are set

5. **Use service references**
   - Define services once in `services.yaml`
   - Reference them in strategy pairs with `$service_name`
   - Enables service reuse across strategies

6. **Version your configurations**
   - Use semantic versioning
   - Document breaking changes
   - Maintain backward compatibility when possible

## Troubleshooting

### Validation Errors

If you get schema validation errors:
1. Check that all required fields are present
2. Verify field types match the schema
3. Ensure service names use valid characters (alphanumeric + underscore)
4. Check that strategy names use valid characters (lowercase + hyphen)

### Service Reference Errors

If service references fail:
1. Verify the service exists in `services.yaml`
2. Check the service name spelling (case-sensitive)
3. Ensure you're using `$service_name` syntax (with dollar sign)

### Environment Variable Errors

If environment variables aren't resolving:
1. Check that variables are set in your environment
2. Verify variable names match exactly (case-sensitive)
3. For `.env` files, ensure they're loaded before validation
4. Check for typos in `${VAR_NAME}` syntax

## Next Steps

- See `docs/configuration/service-registry-schema.md` for complete schema reference
- See `docs/configuration/strategy-pair-schema.md` for strategy pair schema
- See `docs/configuration/environment-variables.md` for environment variable guide
