# Epic 17 Configuration System - Quick Reference

## Installation

```bash
pip install jsonschema python-dotenv
```

## Basic Usage

### 1. Load and Validate Service Registry

```python
from rag_factory.config.validator import load_yaml_with_validation
from rag_factory.config.env_resolver import EnvResolver

# Load services
services = load_yaml_with_validation(
    "config/services.yaml",
    config_type="service_registry"
)

# Resolve environment variables
services = EnvResolver.resolve(services)
```

### 2. Load and Validate Strategy Pair

```python
# Load strategy pair with service reference validation
strategy = load_yaml_with_validation(
    "strategies/semantic-local-pair.yaml",
    config_type="strategy_pair",
    service_registry=services  # For reference validation
)

strategy = EnvResolver.resolve(strategy)
```

## Environment Variable Syntax

### Required Variable
```yaml
api_key: "${API_KEY}"  # Raises error if not set
```

### Optional with Default
```yaml
host: "${DB_HOST:-localhost}"  # Uses 'localhost' if not set
```

### Required with Custom Error
```yaml
password: "${DB_PASSWORD:?Database password is required}"
```

## Service Registry Format

```yaml
version: "1.0.0"

services:
  embedding_local:
    name: "local-onnx-minilm"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    dimensions: 384

  llm_openai:
    name: "openai-gpt-4"
    type: "llm"
    url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"

  db_main:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
```

## Strategy Pair Format

```yaml
strategy_name: "semantic-local-pair"
version: "1.0.0"
description: "Semantic search using local ONNX embeddings"

tags:
  - "semantic"
  - "local"

indexer:
  strategy: "SemanticIndexer"
  services:
    embedding: "$embedding_local"  # Reference to service
    database: "$db_main"
  
  db_config:
    tables:
      chunks: "chunks"
    fields:
      embedding: "embedding"
  
  config:
    chunk_size: 512
    batch_size: 32

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
  tables:
    - "documents"
    - "chunks"
  extensions:
    - "vector"
```

## Validation

### Programmatic Validation

```python
from rag_factory.config.validator import ConfigValidator

validator = ConfigValidator()

# Validate service registry
warnings = validator.validate_services_yaml(config, "services.yaml")

# Validate strategy pair
warnings = validator.validate_strategy_pair_yaml(
    strategy_config,
    service_registry=services,
    file_path="strategy.yaml"
)
```

### Error Handling

```python
from rag_factory.config.validator import ConfigValidationError
from rag_factory.config.env_resolver import EnvironmentVariableError

try:
    config = load_yaml_with_validation("config.yaml", "service_registry")
    config = EnvResolver.resolve(config)
except ConfigValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"File: {e.file_path}")
    print(f"Field: {e.field}")
except EnvironmentVariableError as e:
    print(f"Environment variable error: {e}")
```

## Common Patterns

### Inline Service Configuration

Instead of referencing a service, you can define it inline:

```yaml
indexer:
  strategy: "SemanticIndexer"
  services:
    embedding:
      name: "inline-embedding"
      type: "embedding"
      provider: "onnx"
      model: "test-model"
```

### Multiple Service References

```yaml
indexer:
  strategy: "ComplexIndexer"
  services:
    embedding: "$embedding_local"
    llm: "$llm_openai"
    database: "$db_main"
```

### Database Table Mapping

```yaml
db_config:
  tables:
    documents: "my_documents"      # Logical -> Physical
    chunks: "my_chunks"
  fields:
    chunk_id: "id"
    content: "text_content"
    embedding: "vector_embedding"
```

## Best Practices

1. **Always use environment variables for secrets**
   ```yaml
   api_key: "${OPENAI_API_KEY}"  # ✅ Good
   api_key: "sk-123456"           # ❌ Bad - plaintext secret
   ```

2. **Provide sensible defaults**
   ```yaml
   host: "${DB_HOST:-localhost}"
   port: 5432
   ```

3. **Use service references for reusability**
   ```yaml
   # Define once in services.yaml
   services:
     embedding1:
       name: "my-embedding"
       type: "embedding"
       provider: "onnx"
   
   # Reference multiple times in strategy pairs
   indexer:
     services:
       embedding: "$embedding1"
   retriever:
     services:
       embedding: "$embedding1"
   ```

4. **Document required environment variables**
   Create a `.env.example` file:
   ```bash
   # Required
   DATABASE_URL=postgresql://user:pass@localhost:5432/db
   OPENAI_API_KEY=sk-...
   
   # Optional (with defaults)
   DB_HOST=localhost
   LLM_URL=http://localhost:1234/v1
   ```

## Troubleshooting

### "Service reference not found"
- Check service name spelling (case-sensitive)
- Ensure service exists in `services.yaml`
- Verify you're using `$service_name` syntax (with dollar sign)

### "Required environment variable not set"
- Check variable name spelling
- Ensure variable is set in your environment
- For `.env` files, load them before validation

### "Schema validation failed"
- Check all required fields are present
- Verify field types match the schema
- Ensure names use valid characters:
  - Service names: alphanumeric + underscore
  - Strategy names: lowercase + hyphen

## Examples Location

See `rag_factory/config/examples/` for:
- `services.yaml`: Complete service registry example
- `semantic-local-pair.yaml`: Basic strategy pair
- `hybrid-search-pair.yaml`: Advanced strategy pair
- `README.md`: Detailed documentation

## API Reference

### ConfigValidator

```python
validator = ConfigValidator(schemas_dir=None)
warnings = validator.validate_services_yaml(config, file_path=None)
warnings = validator.validate_strategy_pair_yaml(config, service_registry=None, file_path=None)
```

### EnvResolver

```python
resolved = EnvResolver.resolve(value)  # Recursively resolve env vars
variables = EnvResolver.extract_variable_names(value)  # Extract var names
is_valid = EnvResolver.validate_no_injection(var_name)  # Validate var name
```

### Helper Functions

```python
config = load_yaml_with_validation(file_path, config_type, service_registry=None)
```

## Schema Versions

Current versions:
- Service Registry: `1.0.0`
- Strategy Pair: `1.0.0`

Check compatibility:
```python
from rag_factory.config.schemas import is_compatible

if is_compatible("service_registry", "1.0.0"):
    # Version is compatible
    pass
```
