# Configuration

Learn how to configure RAG Factory for your use case.

---

## Configuration Methods

RAG Factory supports multiple configuration methods:

1. **Python Dictionaries** - Direct configuration in code
2. **YAML Files** - External configuration files
3. **Environment Variables** - System-level configuration
4. **JSON Files** - Alternative file format

---

## Basic Configuration

### Database Configuration

Configure your PostgreSQL database connection:

```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "ragdb",
    "user": "raguser",
    "password": "secure_password",
    "pool_size": 10,
    "pool_timeout": 30
}
```

### Using Environment Variables

```bash
export RAG_DB_HOST=localhost
export RAG_DB_PORT=5432
export RAG_DB_NAME=ragdb
export RAG_DB_USER=raguser
export RAG_DB_PASSWORD=secure_password
```

---

## Strategy Configuration

Each strategy has its own configuration options. Here's an example for Contextual Retrieval:

```python
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

config = ContextualRetrievalConfig(
    chunk_size=512,
    chunk_overlap=50,
    top_k=5,
    context_window=2,
    use_bm25=True,
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)
```

---

## Configuration Files

### YAML Configuration

Create a `config.yaml` file:

```yaml
strategy_name: contextual
chunk_size: 512
chunk_overlap: 50
top_k: 5
context_window: 2
use_bm25: true
llm_provider: openai
llm_model: gpt-3.5-turbo
```

Load it in your code:

```python
from rag_factory.factory import RAGFactory

factory = RAGFactory()
strategy = factory.create_from_config("config.yaml")
```

### JSON Configuration

Alternatively, use JSON:

```json
{
  "strategy_name": "contextual",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 5,
  "context_window": 2,
  "use_bm25": true,
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo"
}
```

---

## LLM Provider Configuration

### OpenAI

```python
llm_config = {
    "provider": "openai",
    "api_key": "your-api-key",  # Or use OPENAI_API_KEY env var
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500
}
```

### Anthropic

```python
llm_config = {
    "provider": "anthropic",
    "api_key": "your-api-key",  # Or use ANTHROPIC_API_KEY env var
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.7,
    "max_tokens": 500
}
```

---

## Embedding Configuration

Configure the embedding model:

```python
from rag_factory.services.embedding_service import EmbeddingService

embedding_service = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",  # or "cpu"
    batch_size=32
)
```

### Using ONNX Models

For faster inference:

```python
embedding_service = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_onnx=True,
    onnx_provider="CUDAExecutionProvider"  # or "CPUExecutionProvider"
)
```

---

## Common Configuration Patterns

### Development Configuration

```python
config = {
    "db_host": "localhost",
    "db_name": "ragdb_dev",
    "chunk_size": 256,
    "top_k": 3,
    "log_level": "DEBUG"
}
```

### Production Configuration

```python
config = {
    "db_host": "prod-db.example.com",
    "db_name": "ragdb_prod",
    "db_pool_size": 20,
    "chunk_size": 512,
    "top_k": 10,
    "use_cache": True,
    "log_level": "INFO"
}
```

---

## Next Steps

- **[Complete Configuration Reference](../guides/configuration-reference.md)** - All configuration options
- **[Strategy-Specific Configuration](../strategies/overview.md)** - Per-strategy settings
- **[Performance Tuning](../guides/performance-tuning.md)** - Optimize your configuration

---

## See Also

- [Environment Variables Reference](../guides/configuration-reference.md#environment-variables)
- [Best Practices](../guides/best-practices.md)
- [Troubleshooting Configuration Issues](../troubleshooting/common-errors.md)
