# LLM Service

Unified interface for Large Language Model providers with support for Anthropic Claude, OpenAI, and local Ollama models.

## Features

- **Multi-Provider Support**: Anthropic, OpenAI, and Ollama
- **Prompt Templates**: Reusable templates with variable substitution
- **Token Counting**: Accurate token counting per provider
- **Cost Tracking**: Automatic cost calculation and tracking
- **Rate Limiting**: Configurable rate limits with automatic backoff
- **Streaming**: Real-time response streaming
- **Statistics**: Comprehensive usage analytics

## Quick Start

```python
from rag_factory.services.llm import LLMService, LLMServiceConfig, Message, MessageRole

# Initialize service
config = LLMServiceConfig(
    provider="anthropic",
    model="claude-sonnet-4.5"
)
service = LLMService(config)

# Generate completion
messages = [Message(role=MessageRole.USER, content="Hello!")]
response = service.complete(messages)
print(response.content)
```

## Supported Providers

### Anthropic Claude
```python
config = LLMServiceConfig(
    provider="anthropic",
    model="claude-sonnet-4.5",  # or claude-3-opus-20240229, claude-3-haiku-20240307
    provider_config={"api_key": "your-key"}  # or set ANTHROPIC_API_KEY env var
)
```

### OpenAI
```python
config = LLMServiceConfig(
    provider="openai",
    model="gpt-4-turbo",  # or gpt-4, gpt-3.5-turbo
    provider_config={"api_key": "your-key"}  # or set OPENAI_API_KEY env var
)
```

### Ollama (Local)
```python
config = LLMServiceConfig(
    provider="ollama",
    model="llama2",  # or any installed Ollama model
    provider_config={"base_url": "http://localhost:11434"}  # optional
)
```

## Using Prompt Templates

```python
from rag_factory.services.llm.prompt_template import CommonTemplates

# RAG Question Answering
messages = CommonTemplates.RAG_QA.format(
    context="Paris is the capital of France.",
    question="What is the capital of France?"
)
response = service.complete(messages)

# Summarization
messages = CommonTemplates.SUMMARIZATION.format(
    text="Long text here...",
    max_words="50"
)
response = service.complete(messages)

# Custom Template
from rag_factory.services.llm.prompt_template import PromptTemplate

template = PromptTemplate(
    system="You are a {role}",
    user="Help me with: {task}"
)
messages = template.format(role="coding assistant", task="Python debugging")
```

## Streaming Responses

```python
for chunk in service.stream(messages):
    if not chunk.is_final:
        print(chunk.content, end="", flush=True)
```

With callback:
```python
def on_chunk(content):
    print(content, end="", flush=True)

list(service.stream(messages, callback=on_chunk))
```

## Cost Tracking

```python
# Estimate cost before making request
estimated_cost = service.estimate_cost(messages, max_completion_tokens=100)

# Make request
response = service.complete(messages)
print(f"Actual cost: ${response.cost:.6f}")

# Get overall statistics
stats = service.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Total requests: {stats['total_requests']}")
print(f"Average latency: {stats['average_latency']:.2f}s")
```

## Rate Limiting

```python
config = LLMServiceConfig(
    provider="anthropic",
    enable_rate_limiting=True,
    rate_limit_config={
        "requests_per_minute": 50,
        "requests_per_second": 2  # optional
    }
)
```

## Error Handling

All providers include automatic retry with exponential backoff (3 attempts):

```python
from tenacity import RetryError

try:
    response = service.complete(messages)
except RetryError as e:
    print(f"Request failed after retries: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Available Common Templates

- `RAG_QA` - Question answering with context
- `SUMMARIZATION` - Text summarization
- `CLASSIFICATION` - Text classification
- `EXTRACTION` - Information extraction
- `REWRITE` - Text rewriting
- `TRANSLATION` - Language translation

## Configuration Options

```python
LLMServiceConfig(
    provider: str = "anthropic",              # Provider name
    model: Optional[str] = None,              # Model name
    provider_config: Dict[str, Any] = {},     # Provider-specific config
    enable_rate_limiting: bool = True,        # Enable rate limiting
    rate_limit_config: Dict[str, Any] = {}    # Rate limit settings
)
```

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Testing

Run unit tests:
```bash
pytest tests/unit/services/llm/ -v
```

Run integration tests (requires API keys):
```bash
pytest tests/integration/services/test_llm_integration.py -v -m integration
```

## Architecture

- `base.py` - Abstract interfaces and data classes
- `service.py` - Main LLM service orchestrator
- `config.py` - Configuration with Pydantic validation
- `prompt_template.py` - Template system
- `token_counter.py` - Token counting utilities
- `providers/` - Provider implementations
  - `anthropic.py` - Anthropic Claude
  - `openai.py` - OpenAI GPT
  - `ollama.py` - Local Ollama models

## Adding a New Provider

1. Create a new provider class inheriting from `ILLMProvider`
2. Implement all required methods
3. Add to the provider map in `service.py`
4. Add tests

See existing providers for examples.
