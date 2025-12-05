# Configuration Examples

This directory contains example configuration files for the RAG Factory Configuration Management System.

## Files

### Base Configuration
- **config.yaml**: Main configuration file with default settings for all strategies
- **config.json**: JSON format example (functionally equivalent to config.yaml)

### Environment-Specific Overrides
- **config.production.yaml**: Production environment overrides
- **config.test.yaml**: Test environment overrides

## Usage

### Loading Configuration

```python
from rag_factory.config import ConfigManager

# Load base configuration
config = ConfigManager()
config.load("examples/configs/config.yaml")

# Access configuration values
log_level = config.get("global_settings.log_level")
strategy_config = config.get_strategy_config("semantic_search")
```

### Environment-Specific Configuration

The configuration system supports environment-specific overrides using the `RAG_ENV` environment variable:

```bash
# Development (default)
export RAG_ENV=development
python your_app.py

# Production
export RAG_ENV=production
python your_app.py

# Test
export RAG_ENV=test
pytest
```

When an environment is set, the system will:
1. Load the base configuration (config.yaml)
2. Look for an environment-specific file (config.{environment}.yaml)
3. Merge the environment-specific settings on top of the base configuration

### Configuration Structure

#### Global Settings
- `environment`: Environment name (development, test, production)
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `cache_enabled`: Enable/disable result caching
- `cache_ttl`: Cache time-to-live in seconds

#### Strategy Configuration
Each strategy can define:
- `chunk_size`: Size of text chunks (1-8192)
- `chunk_overlap`: Overlap between chunks (0-500)
- `top_k`: Number of results to retrieve (1-100)
- `strategy_name`: Strategy identifier
- `metadata`: Additional strategy-specific parameters

#### Pipeline Configuration
- `mode`: Execution mode (sequential, parallel, cascade)
- `timeout`: Pipeline timeout in seconds
- `stages`: List of pipeline stages with their configurations

### Hot-Reload (Development)

Enable automatic configuration reloading when files change:

```python
def on_config_change(new_config):
    print("Configuration reloaded!")

config.enable_hot_reload(callback=on_config_change)
```

## Best Practices

1. **Use environment variables for secrets**: Don't store API keys or passwords in config files
2. **Keep production configs simple**: Avoid complex configurations in production
3. **Test all environments**: Ensure configurations work across dev, test, and prod
4. **Document changes**: Comment any non-obvious configuration choices
5. **Version control**: Track configuration changes in git
6. **Validate early**: Configuration is validated on load, so invalid configs fail fast
