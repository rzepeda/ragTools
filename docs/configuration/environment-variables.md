# Environment Variables Guide

## Overview

The RAG Factory configuration system supports environment variable substitution in YAML files, enabling secure secret management and flexible configuration across different environments.

## Syntax

The configuration system supports three environment variable patterns:

### 1. Required Variable

```yaml
api_key: "${API_KEY}"
```

- **Behavior**: Raises error if variable is not set
- **Use case**: Required configuration that must be provided
- **Error message**: `"Required environment variable ${API_KEY} is not set"`

### 2. Optional Variable with Default

```yaml
host: "${DB_HOST:-localhost}"
```

- **Behavior**: Uses default value if variable is not set
- **Use case**: Optional configuration with sensible defaults
- **Example**: If `DB_HOST` is not set, uses `"localhost"`

### 3. Required Variable with Custom Error

```yaml
api_key: "${API_KEY:?OpenAI API key is required for this service}"
```

- **Behavior**: Raises error with custom message if variable is not set
- **Use case**: Required configuration with helpful error message
- **Error message**: `"Environment variable ${API_KEY}: OpenAI API key is required for this service"`

## Usage Examples

### Basic Usage

```yaml
services:
  llm_openai:
    name: "OpenAI GPT-4"
    type: "llm"
    api_key: "${OPENAI_API_KEY}"
    url: "https://api.openai.com/v1"
    model: "gpt-4"
```

### With Defaults

```yaml
services:
  db_main:
    name: "PostgreSQL Database"
    type: "postgres"
    host: "${DB_HOST:-localhost}"
    port: ${DB_PORT:-5432}
    database: "${DB_NAME:-rag_factory}"
    user: "${DB_USER:-postgres}"
    password: "${DB_PASSWORD}"
```

### Partial String Replacement

```yaml
services:
  db_main:
    name: "PostgreSQL Database"
    type: "postgres"
    connection_string: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
```

### Nested Structures

```yaml
indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"
  config:
    chunk_size: ${CHUNK_SIZE:-512}
    chunk_overlap: ${CHUNK_OVERLAP:-50}
    model_path: "${MODEL_PATH}"
    cache_dir: "${CACHE_DIR:-./models}"
```

## Environment Setup

### Using .env Files

Create a `.env` file in your project root:

```bash
# .env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_factory
DB_USER=postgres
DB_PASSWORD=secret_password

# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Cohere Configuration
COHERE_API_KEY=...

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-...

# Neo4j Configuration
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Model Configuration
MODEL_PATH=/path/to/models
CACHE_DIR=./models/cache
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Loading .env Files

```python
from dotenv import load_dotenv
from rag_factory.config import load_yaml_with_validation, EnvResolver

# Load environment variables from .env file
load_dotenv()

# Load and validate configuration
config = load_yaml_with_validation(
    "config/services.yaml",
    config_type="service_registry"
)

# Resolve environment variables
config = EnvResolver.resolve(config)
```

### System Environment Variables

Set environment variables in your shell:

```bash
# Bash/Zsh
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://..."

# Windows CMD
set OPENAI_API_KEY=sk-...
set DATABASE_URL=postgresql://...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:DATABASE_URL="postgresql://..."
```

### Docker Environment

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-factory:
    image: rag-factory:latest
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/rag_factory
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHUNK_SIZE=512
    env_file:
      - .env
```

### Kubernetes Secrets

```yaml
# kubernetes-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-factory-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  DATABASE_URL: "postgresql://..."

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-factory
spec:
  template:
    spec:
      containers:
      - name: rag-factory
        envFrom:
        - secretRef:
            name: rag-factory-secrets
        env:
        - name: CHUNK_SIZE
          value: "512"
```

## Security Best Practices

### 1. Never Commit Secrets

❌ **Bad**:
```yaml
services:
  llm_openai:
    api_key: "sk-1234567890abcdef"  # NEVER DO THIS!
```

✅ **Good**:
```yaml
services:
  llm_openai:
    api_key: "${OPENAI_API_KEY}"
```

### 2. Use .env.example

Create a `.env.example` file with dummy values:

```bash
# .env.example
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-key-here
```

Add to `.gitignore`:
```
.env
.env.local
*.env
```

### 3. Validate Environment Variables

```python
from rag_factory.config.env_resolver import EnvResolver

# Extract all required variables
config_dict = {...}
required_vars = EnvResolver.extract_variable_names(config_dict)

# Check which are missing
import os
missing = [var for var in required_vars if not os.getenv(var)]

if missing:
    raise ValueError(f"Missing required environment variables: {missing}")
```

### 4. Use Secret Management Services

For production deployments:

- **AWS Secrets Manager**
- **Azure Key Vault**
- **Google Cloud Secret Manager**
- **HashiCorp Vault**

Example with AWS Secrets Manager:

```python
import boto3
import json
import os

def load_secrets_from_aws():
    client = boto3.client('secretsmanager')
    secret = client.get_secret_value(SecretId='rag-factory/prod')
    secrets = json.loads(secret['SecretString'])
    
    for key, value in secrets.items():
        os.environ[key] = value

# Load secrets before loading configuration
load_secrets_from_aws()

# Now load configuration
config = load_yaml_with_validation(...)
```

## Variable Resolution

### Resolution Order

1. Environment variables are resolved recursively
2. Nested structures (dicts, lists) are traversed
3. String values are scanned for `${...}` patterns
4. Variables are replaced with their values

### Example

```python
from rag_factory.config.env_resolver import EnvResolver
import os

# Set environment variables
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'

# Configuration with variables
config = {
    'connection': 'postgresql://${DB_HOST}:${DB_PORT}/db',
    'settings': {
        'host': '${DB_HOST}',
        'port': ${DB_PORT:-5432}
    }
}

# Resolve
resolved = EnvResolver.resolve(config)

# Result:
# {
#     'connection': 'postgresql://localhost:5432/db',
#     'settings': {
#         'host': 'localhost',
#         'port': 5432
#     }
# }
```

### Extracting Variable Names

```python
from rag_factory.config.env_resolver import EnvResolver

config = {
    'api_key': '${OPENAI_API_KEY}',
    'url': '${API_URL:-http://localhost}',
    'connection': 'postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}/db'
}

variables = EnvResolver.extract_variable_names(config)
# ['API_URL', 'DB_HOST', 'DB_PASS', 'DB_USER', 'OPENAI_API_KEY']
```

## Common Patterns

### Database Connection Strings

```yaml
# Full connection string
services:
  db_main:
    connection_string: "${DATABASE_URL}"

# Composed connection string
services:
  db_main:
    connection_string: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# With SSL parameters
services:
  db_main:
    connection_string: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=${DB_SSL_MODE:-require}"
```

### API Keys

```yaml
services:
  # Required API key
  llm_openai:
    api_key: "${OPENAI_API_KEY}"

  # Optional API key with custom error
  llm_anthropic:
    api_key: "${ANTHROPIC_API_KEY:?Anthropic API key required for this service}"
```

### Model Paths

```yaml
services:
  embedding_local:
    model: "${MODEL_NAME:-Xenova/all-MiniLM-L6-v2}"
    cache_dir: "${MODEL_CACHE_DIR:-./models}"
```

### Feature Flags

```yaml
indexer:
  config:
    enable_caching: ${ENABLE_CACHE:-true}
    debug_mode: ${DEBUG_MODE:-false}
    max_workers: ${MAX_WORKERS:-4}
```

## Troubleshooting

### Common Errors

#### 1. Variable Not Set

```
EnvironmentVariableError: Required environment variable ${DATABASE_URL} is not set
```

**Solutions**:
- Set the environment variable: `export DATABASE_URL="..."`
- Add to `.env` file
- Provide a default: `${DATABASE_URL:-postgresql://localhost/db}`

#### 2. Variable Name Typo

```
EnvironmentVariableError: Required environment variable ${DATABSE_URL} is not set
```

**Solution**: Fix the typo in the configuration file

#### 3. Syntax Error

```yaml
# Wrong
api_key: ${API_KEY}  # Missing quotes and braces

# Correct
api_key: "${API_KEY}"
```

#### 4. Default Value Not Working

```yaml
# Wrong - missing colon before dash
host: "${DB_HOST-localhost}"

# Correct
host: "${DB_HOST:-localhost}"
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from rag_factory.config.env_resolver import EnvResolver

# This will show which variables are being resolved
config = EnvResolver.resolve(config_dict)
```

Check which variables are required:

```python
from rag_factory.config.env_resolver import EnvResolver
import yaml

with open('config/services.yaml') as f:
    config = yaml.safe_load(f)

# Get all variable names
variables = EnvResolver.extract_variable_names(config)
print(f"Required variables: {variables}")

# Check which are set
import os
for var in variables:
    value = os.getenv(var)
    if value:
        print(f"✓ {var} is set")
    else:
        print(f"✗ {var} is NOT set")
```

## Best Practices

1. **Use descriptive variable names**: `OPENAI_API_KEY` not `KEY1`
2. **Group related variables**: Use prefixes like `DB_*`, `OPENAI_*`
3. **Provide defaults for development**: Use `:-` syntax for local development
4. **Document all variables**: Maintain `.env.example` file
5. **Validate before deployment**: Check all required variables are set
6. **Use secret management**: Don't rely on `.env` files in production
7. **Never log resolved secrets**: Log variable names, not values
8. **Rotate secrets regularly**: Update API keys and passwords periodically
9. **Use different values per environment**: dev, staging, production
10. **Test with missing variables**: Ensure error messages are helpful

## Example .env.example

```bash
# .env.example
# Copy this file to .env and fill in your values

# ============================================================================
# Database Configuration
# ============================================================================
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_factory
DB_USER=postgres
DB_PASSWORD=your_password_here

# ============================================================================
# LLM API Keys
# ============================================================================
# OpenAI (required for OpenAI services)
OPENAI_API_KEY=sk-your-key-here

# Cohere (required for Cohere services)
COHERE_API_KEY=your-key-here

# Anthropic (required for Claude services)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# ============================================================================
# Graph Database (optional, for knowledge graph strategies)
# ============================================================================
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_PATH=/path/to/models
CACHE_DIR=./models/cache
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# ============================================================================
# Feature Flags
# ============================================================================
ENABLE_CACHE=true
DEBUG_MODE=false
MAX_WORKERS=4
```

## References

- [Service Registry Schema](service-registry-schema.md)
- [Strategy Pair Schema](strategy-pair-schema.md)
- [Configuration Examples](../../rag_factory/config/examples/)
