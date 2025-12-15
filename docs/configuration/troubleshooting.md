# Configuration Troubleshooting Guide

## Overview

This guide helps you diagnose and fix common issues with RAG Factory configuration files.

## Quick Diagnostics

### Validate Your Configuration

```python
from rag_factory.config.validator import load_yaml_with_validation
from rag_factory.config.env_resolver import EnvResolver
import yaml

try:
    # Load service registry
    services = load_yaml_with_validation(
        "config/services.yaml",
        config_type="service_registry"
    )
    print("✓ Service registry is valid")
    
    # Resolve environment variables
    services = EnvResolver.resolve(services)
    print("✓ Environment variables resolved")
    
    # Load strategy pair
    strategy = load_yaml_with_validation(
        "config/strategy-pair.yaml",
        config_type="strategy_pair",
        service_registry=services
    )
    print("✓ Strategy pair is valid")
    
    strategy = EnvResolver.resolve(strategy)
    print("✓ All configurations valid!")
    
except Exception as e:
    print(f"✗ Error: {e}")
```

## Common Errors

### Schema Validation Errors

#### Error: "Schema validation failed: 'name' is a required property"

**Cause**: Missing required field in service configuration

**Solution**: Add the missing field

```yaml
# ❌ Wrong
services:
  llm1:
    type: "llm"
    url: "http://localhost:1234/v1"

# ✅ Correct
services:
  llm1:
    name: "Local LLM"  # Added required field
    type: "llm"
    url: "http://localhost:1234/v1"
```

#### Error: "Schema validation failed: 'xyz' is not one of ['onnx', 'openai', 'cohere', 'huggingface']"

**Cause**: Invalid enum value

**Solution**: Use a valid provider value

```yaml
# ❌ Wrong
services:
  embedding1:
    name: "My Embedding"
    type: "embedding"
    provider: "custom"  # Invalid

# ✅ Correct
services:
  embedding1:
    name: "My Embedding"
    type: "embedding"
    provider: "onnx"  # Valid enum value
```

#### Error: "Schema validation failed: Additional properties are not allowed"

**Cause**: Unknown field in configuration

**Solution**: Remove the unknown field or check for typos

```yaml
# ❌ Wrong
services:
  llm1:
    name: "LLM"
    type: "llm"
    api_key: "${API_KEY}"
    unknown_field: "value"  # Not in schema

# ✅ Correct
services:
  llm1:
    name: "LLM"
    type: "llm"
    api_key: "${API_KEY}"
```

#### Error: "Schema validation failed: 'strategy-name_123' does not match '^[a-z0-9-]+$'"

**Cause**: Invalid characters in strategy name

**Solution**: Use only lowercase letters, numbers, and hyphens

```yaml
# ❌ Wrong
strategy_name: "Strategy_Name_123"  # Uppercase and underscores

# ✅ Correct
strategy_name: "strategy-name-123"  # Lowercase and hyphens
```

### Service Reference Errors

#### Error: "Service reference '$nonexistent' not found in registry"

**Cause**: Referenced service doesn't exist in service registry

**Solution**: Add the service to `services.yaml` or fix the reference

```yaml
# services.yaml
services:
  embedding_local:  # Make sure this exists
    name: "Local Embedding"
    type: "embedding"
    provider: "onnx"

# strategy-pair.yaml
indexer:
  services:
    embedding: "$embedding_local"  # Reference must match exactly
```

#### Error: "Service reference 'embedding_local' not found"

**Cause**: Missing `$` prefix in service reference

**Solution**: Add `$` prefix

```yaml
# ❌ Wrong
indexer:
  services:
    embedding: "embedding_local"  # Missing $

# ✅ Correct
indexer:
  services:
    embedding: "$embedding_local"  # With $
```

### Environment Variable Errors

#### Error: "Required environment variable ${DATABASE_URL} is not set"

**Cause**: Environment variable is not set

**Solutions**:

1. Set the environment variable:
```bash
export DATABASE_URL="postgresql://localhost/db"
```

2. Add to `.env` file:
```bash
# .env
DATABASE_URL=postgresql://localhost/db
```

3. Provide a default value:
```yaml
connection_string: "${DATABASE_URL:-postgresql://localhost/db}"
```

#### Error: "Environment variable ${API_KEY}: Custom error message"

**Cause**: Required variable with custom error is not set

**Solution**: Set the environment variable

```bash
export API_KEY="your-key-here"
```

#### Error: Syntax error in environment variable

**Cause**: Incorrect syntax

**Solution**: Use correct syntax

```yaml
# ❌ Wrong
api_key: ${API_KEY}          # Missing quotes and braces
api_key: "$API_KEY"          # Missing braces
api_key: "${API_KEY}"        # Missing closing brace

# ✅ Correct
api_key: "${API_KEY}"
api_key: "${API_KEY:-default}"
api_key: "${API_KEY:?Error message}"
```

### YAML Parsing Errors

#### Error: "yaml.scanner.ScannerError: mapping values are not allowed here"

**Cause**: YAML syntax error, often missing quotes

**Solution**: Add quotes around values with special characters

```yaml
# ❌ Wrong
connection_string: postgresql://user:pass@host:5432/db

# ✅ Correct
connection_string: "postgresql://user:pass@host:5432/db"
```

#### Error: "yaml.parser.ParserError: expected <block end>, but found"

**Cause**: Indentation error

**Solution**: Fix indentation (use 2 or 4 spaces consistently)

```yaml
# ❌ Wrong
services:
  llm1:
  name: "LLM"  # Wrong indentation

# ✅ Correct
services:
  llm1:
    name: "LLM"  # Correct indentation
```

### File Not Found Errors

#### Error: "FileNotFoundError: [Errno 2] No such file or directory: 'config/services.yaml'"

**Cause**: Configuration file doesn't exist or wrong path

**Solutions**:

1. Check file exists:
```bash
ls -la config/services.yaml
```

2. Use absolute path:
```python
import os
config_path = os.path.join(os.getcwd(), "config", "services.yaml")
```

3. Create the file if missing:
```bash
mkdir -p config
touch config/services.yaml
```

## Warnings

### Warning: "Potential plaintext secret in services.llm1.api_key"

**Cause**: API key appears to be plaintext instead of environment variable

**Solution**: Use environment variable

```yaml
# ❌ Triggers warning
services:
  llm1:
    api_key: "sk-1234567890"  # Plaintext

# ✅ No warning
services:
  llm1:
    api_key: "${OPENAI_API_KEY}"  # Environment variable
```

## Debugging Techniques

### 1. Validate Schema Step by Step

```python
from rag_factory.config.validator import ConfigValidator
import yaml

validator = ConfigValidator()

# Load YAML
with open("config/services.yaml") as f:
    config = yaml.safe_load(f)

# Validate
try:
    warnings = validator.validate_services_yaml(config, "config/services.yaml")
    print(f"Validation successful with {len(warnings)} warnings")
    for warning in warnings:
        print(f"  - {warning}")
except Exception as e:
    print(f"Validation failed: {e}")
```

### 2. Check Environment Variables

```python
from rag_factory.config.env_resolver import EnvResolver
import os

# Extract required variables
config = {...}
variables = EnvResolver.extract_variable_names(config)

print("Required environment variables:")
for var in variables:
    value = os.getenv(var)
    if value:
        print(f"  ✓ {var} = {value[:10]}..." if len(value) > 10 else f"  ✓ {var} = {value}")
    else:
        print(f"  ✗ {var} is NOT set")
```

### 3. Test Resolution

```python
from rag_factory.config.env_resolver import EnvResolver
import os

# Set test variables
os.environ['TEST_VAR'] = 'test_value'

# Test resolution
test_config = {
    'field1': '${TEST_VAR}',
    'field2': '${MISSING_VAR:-default}',
    'field3': 'static_value'
}

try:
    resolved = EnvResolver.resolve(test_config)
    print("Resolution successful:")
    print(resolved)
except Exception as e:
    print(f"Resolution failed: {e}")
```

### 4. Validate Service References

```python
from rag_factory.config.validator import ConfigValidator
import yaml

# Load both files
with open("config/services.yaml") as f:
    services = yaml.safe_load(f)

with open("config/strategy-pair.yaml") as f:
    strategy = yaml.safe_load(f)

# Validate references
validator = ConfigValidator()
try:
    warnings = validator.validate_strategy_pair_yaml(
        strategy,
        service_registry=services,
        file_path="config/strategy-pair.yaml"
    )
    print(f"Service references valid with {len(warnings)} warnings")
except Exception as e:
    print(f"Service reference validation failed: {e}")
```

## Best Practices for Avoiding Issues

### 1. Use a Linter

Install and use a YAML linter:

```bash
pip install yamllint

# Create .yamllint config
cat > .yamllint << EOF
extends: default
rules:
  line-length:
    max: 120
  indentation:
    spaces: 2
EOF

# Lint your files
yamllint config/
```

### 2. Validate Before Committing

Add a pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit

python -c "
from rag_factory.config.validator import load_yaml_with_validation
import sys

try:
    load_yaml_with_validation('config/services.yaml', 'service_registry')
    print('✓ Configuration valid')
except Exception as e:
    print(f'✗ Configuration invalid: {e}')
    sys.exit(1)
"
```

### 3. Use Schema Validation in CI/CD

```yaml
# .github/workflows/validate-config.yml
name: Validate Configuration

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -e .
      - name: Validate configuration
        run: python scripts/validate_config.py
```

### 4. Document Required Variables

Maintain `.env.example`:

```bash
# .env.example
# Required variables (no defaults)
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://user:pass@host/db

# Optional variables (have defaults)
DB_HOST=localhost
CHUNK_SIZE=512
```

### 5. Test in Isolation

Create test configurations:

```yaml
# config/test/services.yaml
services:
  test_embedding:
    name: "Test Embedding"
    type: "embedding"
    provider: "onnx"
    model: "test-model"
```

## Common Patterns and Solutions

### Pattern: Multiple Environments

Use environment-specific configuration files:

```
config/
  ├── services.yaml              # Base configuration
  ├── services.dev.yaml          # Development overrides
  ├── services.staging.yaml      # Staging overrides
  └── services.prod.yaml         # Production overrides
```

Load based on environment:

```python
import os
from rag_factory.config.validator import load_yaml_with_validation

env = os.getenv('ENVIRONMENT', 'dev')
config_file = f"config/services.{env}.yaml"

services = load_yaml_with_validation(config_file, "service_registry")
```

### Pattern: Shared Services

Define common services once:

```yaml
# config/services.common.yaml
services:
  embedding_local:
    name: "Local Embedding"
    type: "embedding"
    provider: "onnx"

# config/services.yaml
services:
  # Include common services
  embedding_local:
    $ref: "services.common.yaml#/services/embedding_local"
```

### Pattern: Conditional Configuration

Use environment variables for feature flags:

```yaml
indexer:
  config:
    enable_caching: ${ENABLE_CACHE:-true}
    debug_mode: ${DEBUG_MODE:-false}
    use_gpu: ${USE_GPU:-false}
```

## Getting Help

### Check Logs

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validate JSON Schema

Check the schema files directly:

```bash
# View service registry schema
cat rag_factory/config/schemas/service_registry_schema.json | jq

# View strategy pair schema
cat rag_factory/config/schemas/strategy_pair_schema.json | jq
```

### Use Examples

Copy and modify example configurations:

```bash
cp rag_factory/config/examples/services.yaml config/services.yaml
cp rag_factory/config/examples/semantic-local-pair.yaml config/my-pair.yaml
```

### Community Resources

- Documentation: `docs/configuration/`
- Examples: `rag_factory/config/examples/`
- Tests: `tests/unit/config/` and `tests/integration/config/`
- GitHub Issues: Report bugs and ask questions

## Checklist for Troubleshooting

- [ ] YAML syntax is valid (no tabs, correct indentation)
- [ ] All required fields are present
- [ ] Field types match schema (string, integer, etc.)
- [ ] Enum values are valid (provider, type, etc.)
- [ ] Service names use valid characters (alphanumeric + underscore)
- [ ] Strategy names use valid characters (lowercase + hyphen)
- [ ] Service references use `$` prefix
- [ ] Referenced services exist in registry
- [ ] Environment variables are set
- [ ] Environment variable syntax is correct
- [ ] No plaintext secrets in configuration
- [ ] File paths are correct
- [ ] Schema files exist and are valid
- [ ] Dependencies are installed (pyyaml, jsonschema, python-dotenv)

## Quick Reference

### Validation Command

```bash
python -c "
from rag_factory.config.validator import load_yaml_with_validation
config = load_yaml_with_validation('config/services.yaml', 'service_registry')
print('✓ Valid')
"
```

### Environment Variable Check

```bash
python -c "
from rag_factory.config.env_resolver import EnvResolver
import yaml
with open('config/services.yaml') as f:
    config = yaml.safe_load(f)
vars = EnvResolver.extract_variable_names(config)
print('Required variables:', vars)
"
```

### Service Reference Check

```bash
python -c "
import yaml
with open('config/services.yaml') as f:
    services = yaml.safe_load(f)
print('Available services:', list(services['services'].keys()))
"
```
