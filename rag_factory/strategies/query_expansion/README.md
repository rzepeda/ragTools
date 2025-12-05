# Query Expansion Strategy

LLM-based query expansion to improve search precision and recall.

## Overview

The query expansion module enhances user queries by adding relevant keywords, reformulating questions, or generating hypothetical documents. This improves retrieval quality in RAG systems by making queries more specific and comprehensive.

## Features

- **Multiple Expansion Strategies**
  - Keyword expansion
  - Query reformulation
  - Question generation
  - Multi-query variants
  - HyDE (Hypothetical Document Expansion)

- **Performance Optimizations**
  - In-memory caching with TTL
  - Configurable timeout handling
  - Graceful error fallback

- **Observability**
  - Execution time tracking
  - Token usage monitoring
  - Cost tracking
  - A/B testing support
  - Comprehensive metrics

## Quick Start

```python
from rag_factory.strategies.query_expansion import (
    QueryExpanderService,
    ExpansionConfig,
    ExpansionStrategy
)
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig

# Initialize LLM service
llm_config = LLMServiceConfig(
    provider="openai",
    model="gpt-3.5-turbo"
)
llm_service = LLMService(llm_config)

# Configure expansion
config = ExpansionConfig(
    strategy=ExpansionStrategy.KEYWORD,
    max_additional_terms=5,
    enable_cache=True
)

# Create service
service = QueryExpanderService(config, llm_service)

# Expand query
result = service.expand("machine learning")
print(f"Expanded: {result.primary_expansion.expanded_query}")
```

## Expansion Strategies

### 1. Keyword Expansion

Adds relevant keywords and synonyms to the original query.

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.KEYWORD,
    max_additional_terms=5
)
```

**Example:**
- Input: "machine learning"
- Output: "machine learning algorithms neural networks deep learning AI"

### 2. Query Reformulation

Rephrases the query to be more specific and searchable.

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.REFORMULATION
)
```

**Example:**
- Input: "how does it work"
- Output: "how does machine learning algorithm training work"

### 3. Question Generation

Converts queries into well-formed questions.

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.QUESTION_GENERATION
)
```

**Example:**
- Input: "python tutorial"
- Output: "What is a good Python programming tutorial for beginners?"

### 4. Multi-Query

Generates multiple variations of the query to improve coverage.

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.MULTI_QUERY,
    generate_multiple_variants=True,
    num_variants=3
)
```

**Example:**
- Input: "climate change effects"
- Outputs:
  1. "What are the environmental effects of climate change?"
  2. "How does climate change impact ecosystems?"
  3. "What are the consequences of global warming?"

### 5. HyDE (Hypothetical Document Expansion)

Generates a hypothetical document that would answer the query, then uses it for retrieval.

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.HYDE,
    max_tokens=150
)
```

**Example:**
- Input: "What is the capital of France?"
- Output: "The capital of France is Paris, a major European city located in the north-central part of the country. Paris has been France's capital since..."

## Configuration Options

### Basic Settings

```python
config = ExpansionConfig(
    # Strategy selection
    strategy=ExpansionStrategy.KEYWORD,

    # LLM settings
    llm_model="gpt-3.5-turbo",
    max_tokens=150,
    temperature=0.3,  # Lower for consistency

    # Expansion control
    max_additional_terms=5,
    generate_multiple_variants=False,
    num_variants=3,

    # Performance
    enable_cache=True,
    cache_ttl=3600,  # seconds
    timeout_seconds=5.0,

    # A/B testing
    enable_expansion=True,
    track_metrics=True
)
```

### Domain-Specific Expansion

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.KEYWORD,
    domain_context="medical and healthcare context",
    max_additional_terms=5
)
```

### Custom Prompts

```python
config = ExpansionConfig(
    strategy=ExpansionStrategy.KEYWORD,
    system_prompt="You are an expert in expanding search queries for academic research...",
)
```

## Caching

The service includes built-in caching to avoid redundant LLM calls:

```python
# First call - cache miss
result1 = service.expand("query")  # ~500ms

# Second call - cache hit
result2 = service.expand("query")  # ~5ms
```

Clear cache when needed:

```python
service.clear_cache()
```

## A/B Testing

Test expansion effectiveness by enabling/disabling per request:

```python
# Control group - no expansion
result_control = service.expand("query", enable_expansion=False)

# Treatment group - with expansion
result_treatment = service.expand("query", enable_expansion=True)

# Compare results
stats = service.get_stats()
print(f"Expansion rate: {stats['expansion_rate']:.2%}")
```

## Metrics and Monitoring

Track performance and quality:

```python
# Get service statistics
stats = service.get_stats()

print(f"Total expansions: {stats['total_expansions']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")
print(f"Expansion rate: {stats['expansion_rate']:.2%}")
```

Access detailed metrics:

```python
# Per-expansion metadata
result = service.expand("query")
metadata = result.primary_expansion.metadata

print(f"Tokens used: {metadata['total_tokens']}")
print(f"Cost: ${metadata['cost']:.6f}")
print(f"Cache hit: {result.cache_hit}")
```

## Error Handling

The service gracefully falls back to the original query on errors:

```python
try:
    result = service.expand("query")

    if "error" in result.metadata:
        print(f"Expansion failed, using original query")
        print(f"Error: {result.metadata['error']}")

    # Original query is always available
    query_to_use = result.primary_expansion.expanded_query

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Choose the Right Strategy**
   - Use `KEYWORD` for general query enhancement
   - Use `HYDE` for semantic search with embeddings
   - Use `MULTI_QUERY` for comprehensive coverage
   - Use `REFORMULATION` for vague queries

2. **Enable Caching**
   - Always enable caching in production
   - Set appropriate TTL based on query patterns
   - Clear cache when updating prompts or strategies

3. **Monitor Performance**
   - Track execution times
   - Monitor cache hit rates
   - Watch token usage and costs

4. **Use A/B Testing**
   - Validate expansion improves results
   - Test different strategies
   - Measure impact on precision/recall

5. **Domain Customization**
   - Provide domain context for specialized use cases
   - Customize system prompts when needed
   - Adjust temperature for consistency

6. **Error Handling**
   - Always handle potential LLM failures
   - Use fallback to original query
   - Set appropriate timeouts

## Architecture

```
QueryExpanderService
├── LLMQueryExpander (strategies: keyword, reformulation, question, multi-query)
├── HyDEExpander (strategy: hyde)
├── ExpansionCache (optional)
└── MetricsTracker (optional)
```

## Testing

Run unit tests:

```bash
pytest tests/unit/strategies/query_expansion/ -v
```

Run integration tests (requires API keys):

```bash
export OPENAI_API_KEY=your_key
pytest tests/integration/strategies/test_query_expansion_integration.py -v
```

## Examples

See `examples/query_expansion_example.py` for comprehensive usage examples:

```bash
python examples/query_expansion_example.py
```

## Performance

### Benchmarks

- **First call:** 200-800ms (depends on LLM provider)
- **Cached call:** <10ms
- **Target:** <1 second per expansion ✅

### Resource Usage

- **Memory:** Minimal (cache is bounded)
- **Network:** One LLM API call per unique query
- **Cost:** Varies by provider and strategy

## API Reference

### Classes

- `ExpansionStrategy` - Enum of available strategies
- `ExpansionConfig` - Configuration dataclass
- `ExpandedQuery` - Single expansion result
- `ExpansionResult` - Complete service result
- `QueryExpanderService` - Main service class
- `ExpansionCache` - Caching implementation
- `MetricsTracker` - Metrics collection

### Key Methods

```python
# Expand a query
result = service.expand(query: str, enable_expansion: bool = None) -> ExpansionResult

# Get statistics
stats = service.get_stats() -> Dict[str, Any]

# Clear cache
service.clear_cache() -> None
```

## Troubleshooting

### Common Issues

1. **Slow expansions**
   - Enable caching
   - Reduce `max_tokens`
   - Use faster LLM model

2. **Poor expansion quality**
   - Adjust `temperature` (lower for consistency)
   - Add `domain_context`
   - Customize `system_prompt`
   - Try different strategy

3. **High costs**
   - Enable caching
   - Reduce `max_tokens`
   - Use cheaper model (gpt-3.5-turbo)
   - Implement request throttling

4. **Cache not working**
   - Verify `enable_cache=True`
   - Check cache TTL
   - Ensure query strings are identical

## License

Part of the RAG Factory project.
