# Query Expansion Integration Guide

This guide shows how to integrate query expansion into your RAG pipeline.

## Table of Contents

1. [Basic Integration](#basic-integration)
2. [Pipeline Integration](#pipeline-integration)
3. [Production Setup](#production-setup)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

---

## Basic Integration

### Step 1: Initialize Services

```python
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig
from rag_factory.strategies.query_expansion import (
    QueryExpanderService,
    ExpansionConfig,
    ExpansionStrategy
)

# Initialize LLM service
llm_config = LLMServiceConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    provider_config={
        "api_key": "your-api-key"
    }
)
llm_service = LLMService(llm_config)

# Configure query expansion
expansion_config = ExpansionConfig(
    strategy=ExpansionStrategy.KEYWORD,
    max_additional_terms=5,
    enable_cache=True,
    track_metrics=True
)

# Create expander service
expander = QueryExpanderService(expansion_config, llm_service)
```

### Step 2: Expand Query Before Retrieval

```python
def retrieve_with_expansion(query: str, retriever, top_k: int = 10):
    """Retrieve documents using expanded query."""

    # Expand the query
    expansion_result = expander.expand(query)

    # Use expanded query for retrieval
    expanded_query = expansion_result.primary_expansion.expanded_query

    # Retrieve documents
    documents = retriever.retrieve(expanded_query, top_k=top_k)

    # Return results with expansion metadata
    return {
        "documents": documents,
        "original_query": expansion_result.original_query,
        "expanded_query": expanded_query,
        "added_terms": expansion_result.primary_expansion.added_terms,
        "expansion_time_ms": expansion_result.execution_time_ms
    }
```

### Step 3: Use in Your Application

```python
# User query
user_query = "machine learning tutorial"

# Retrieve with expansion
results = retrieve_with_expansion(user_query, my_retriever, top_k=10)

print(f"Original: {results['original_query']}")
print(f"Expanded: {results['expanded_query']}")
print(f"Found {len(results['documents'])} documents")
```

---

## Pipeline Integration

### Complete RAG Pipeline with Query Expansion

```python
from rag_factory.strategies.query_expansion import (
    QueryExpanderService,
    ExpansionConfig,
    ExpansionStrategy
)
from rag_factory.services.embedding.service import EmbeddingService
from rag_factory.repositories.chunk import ChunkRepository

class RAGPipelineWithExpansion:
    """Complete RAG pipeline with query expansion."""

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkRepository
    ):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository

        # Initialize query expander
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            max_additional_terms=5,
            enable_cache=True,
            track_metrics=True
        )
        self.expander = QueryExpanderService(expansion_config, llm_service)

    def query(self, user_query: str, top_k: int = 10) -> Dict[str, Any]:
        """Execute complete RAG pipeline with query expansion."""

        # Step 1: Expand query
        expansion_result = self.expander.expand(user_query)
        expanded_query = expansion_result.primary_expansion.expanded_query

        # Step 2: Generate embedding for expanded query
        query_embedding = self.embedding_service.embed_query(expanded_query)

        # Step 3: Retrieve similar chunks
        chunks = self.chunk_repository.search_similar(
            query_embedding,
            top_k=top_k
        )

        # Step 4: Generate answer with LLM
        context = "\n\n".join([chunk.content for chunk in chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

        from rag_factory.services.llm.base import Message, MessageRole
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm_service.complete(messages)

        # Step 5: Return results with metadata
        return {
            "answer": response.content,
            "sources": chunks,
            "original_query": user_query,
            "expanded_query": expanded_query,
            "added_terms": expansion_result.primary_expansion.added_terms,
            "expansion_time_ms": expansion_result.execution_time_ms,
            "total_tokens": response.total_tokens,
            "cost": response.cost
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "expansion_stats": self.expander.get_stats(),
            "embedding_stats": self.embedding_service.get_stats(),
            "llm_stats": self.llm_service.get_stats()
        }
```

### Usage

```python
# Initialize pipeline
pipeline = RAGPipelineWithExpansion(
    llm_service=llm_service,
    embedding_service=embedding_service,
    chunk_repository=chunk_repo
)

# Execute query
result = pipeline.query("What is machine learning?", top_k=10)

print(f"Answer: {result['answer']}")
print(f"Query was expanded: {result['original_query']} -> {result['expanded_query']}")
print(f"Used {len(result['sources'])} sources")
```

---

## Production Setup

### 1. Configuration Management

```python
# config.py
from dataclasses import dataclass
from rag_factory.strategies.query_expansion import ExpansionConfig, ExpansionStrategy

@dataclass
class ProductionConfig:
    """Production configuration for query expansion."""

    # Query expansion settings
    expansion_enabled: bool = True
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.KEYWORD
    max_additional_terms: int = 5
    expansion_cache_enabled: bool = True
    expansion_cache_ttl: int = 3600  # 1 hour

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 150

    # A/B testing
    expansion_rollout_percentage: float = 0.5  # 50% of requests

def get_expansion_config(config: ProductionConfig) -> ExpansionConfig:
    """Create expansion config from production config."""
    return ExpansionConfig(
        strategy=config.expansion_strategy,
        max_additional_terms=config.max_additional_terms,
        llm_model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        enable_cache=config.expansion_cache_enabled,
        cache_ttl=config.expansion_cache_ttl,
        enable_expansion=config.expansion_enabled,
        track_metrics=True
    )
```

### 2. A/B Testing Implementation

```python
import random

class ABTestingRAGPipeline:
    """RAG pipeline with A/B testing for query expansion."""

    def __init__(
        self,
        expander: QueryExpanderService,
        retriever,
        expansion_rollout: float = 0.5
    ):
        self.expander = expander
        self.retriever = retriever
        self.expansion_rollout = expansion_rollout

    def query(self, user_query: str, user_id: str = None) -> Dict[str, Any]:
        """Execute query with A/B test."""

        # Determine if this user gets expansion
        # Use user_id for consistent experience, or random for each request
        if user_id:
            enable_expansion = hash(user_id) % 100 < (self.expansion_rollout * 100)
        else:
            enable_expansion = random.random() < self.expansion_rollout

        # Expand query (or pass through)
        expansion_result = self.expander.expand(
            user_query,
            enable_expansion=enable_expansion
        )

        # Use expanded query for retrieval
        query_text = expansion_result.primary_expansion.expanded_query

        # Retrieve and return
        documents = self.retriever.retrieve(query_text, top_k=10)

        return {
            "documents": documents,
            "original_query": user_query,
            "expanded_query": query_text,
            "expansion_enabled": enable_expansion,
            "expansion_group": "treatment" if enable_expansion else "control"
        }
```

### 3. Monitoring and Logging

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MonitoredQueryExpander:
    """Query expander with comprehensive monitoring."""

    def __init__(self, expander: QueryExpanderService):
        self.expander = expander

    def expand(self, query: str, request_id: str = None) -> ExpansionResult:
        """Expand query with monitoring."""

        start_time = time.time()

        try:
            # Expand query
            result = self.expander.expand(query)

            # Log success
            logger.info(
                "Query expansion successful",
                extra={
                    "request_id": request_id,
                    "original_query": query,
                    "expanded_query": result.primary_expansion.expanded_query,
                    "added_terms_count": len(result.primary_expansion.added_terms),
                    "execution_time_ms": result.execution_time_ms,
                    "cache_hit": result.cache_hit,
                    "strategy": result.primary_expansion.expansion_strategy.value
                }
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                "Query expansion failed",
                extra={
                    "request_id": request_id,
                    "query": query,
                    "error": str(e)
                },
                exc_info=True
            )

            # Re-raise
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get expansion metrics."""
        stats = self.expander.get_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_expansions": stats["total_expansions"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "avg_execution_time_ms": stats["avg_execution_time_ms"],
            "expansion_rate": stats["expansion_rate"]
        }
```

### 4. Error Handling and Fallback

```python
class RobustQueryExpander:
    """Query expander with robust error handling."""

    def __init__(
        self,
        expander: QueryExpanderService,
        max_retries: int = 2,
        fallback_to_original: bool = True
    ):
        self.expander = expander
        self.max_retries = max_retries
        self.fallback_to_original = fallback_to_original

    def expand(self, query: str) -> str:
        """Expand query with retries and fallback."""

        for attempt in range(self.max_retries):
            try:
                result = self.expander.expand(query)
                return result.primary_expansion.expanded_query

            except Exception as e:
                logger.warning(
                    f"Query expansion attempt {attempt + 1} failed: {e}"
                )

                if attempt == self.max_retries - 1:
                    if self.fallback_to_original:
                        logger.info("Falling back to original query")
                        return query
                    else:
                        raise

                # Wait before retry
                time.sleep(0.5 * (attempt + 1))

        return query
```

---

## Advanced Usage

### Multi-Strategy Expansion

```python
class MultiStrategyExpander:
    """Use multiple expansion strategies and combine results."""

    def __init__(self, llm_service: LLMService):
        self.expanders = {
            "keyword": QueryExpanderService(
                ExpansionConfig(strategy=ExpansionStrategy.KEYWORD),
                llm_service
            ),
            "hyde": QueryExpanderService(
                ExpansionConfig(strategy=ExpansionStrategy.HYDE),
                llm_service
            )
        }

    def expand(self, query: str) -> Dict[str, str]:
        """Expand using multiple strategies."""

        results = {}

        for name, expander in self.expanders.items():
            try:
                result = expander.expand(query)
                results[name] = result.primary_expansion.expanded_query
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                results[name] = query

        return results

    def expand_and_retrieve(self, query: str, retriever) -> List[Chunk]:
        """Expand with multiple strategies and combine results."""

        # Get expansions
        expansions = self.expand(query)

        # Retrieve with each expansion
        all_chunks = []
        seen_ids = set()

        for strategy_name, expanded_query in expansions.items():
            chunks = retriever.retrieve(expanded_query, top_k=10)

            # Deduplicate
            for chunk in chunks:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        # Sort by relevance score
        all_chunks.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        return all_chunks[:10]  # Top 10 overall
```

### Adaptive Expansion

```python
class AdaptiveExpander:
    """Automatically select best expansion strategy based on query."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.expanders = {}

    def analyze_query(self, query: str) -> ExpansionStrategy:
        """Analyze query and select best strategy."""

        query_lower = query.lower()

        # Question queries -> question generation
        if any(q in query_lower for q in ["what", "how", "why", "when", "where"]):
            return ExpansionStrategy.QUESTION_GENERATION

        # Very short queries -> keyword expansion
        if len(query.split()) <= 3:
            return ExpansionStrategy.KEYWORD

        # Complex queries -> HyDE
        if len(query.split()) > 10:
            return ExpansionStrategy.HYDE

        # Default
        return ExpansionStrategy.REFORMULATION

    def expand(self, query: str) -> ExpansionResult:
        """Adaptively expand query."""

        # Select strategy
        strategy = self.analyze_query(query)

        # Get or create expander for this strategy
        if strategy not in self.expanders:
            config = ExpansionConfig(strategy=strategy, enable_cache=True)
            self.expanders[strategy] = QueryExpanderService(config, self.llm_service)

        # Expand
        return self.expanders[strategy].expand(query)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Slow Query Expansion

**Problem:** Expansion takes >1 second

**Solutions:**
```python
# Enable caching
config = ExpansionConfig(
    enable_cache=True,
    cache_ttl=3600
)

# Use faster model
config = ExpansionConfig(
    llm_model="gpt-3.5-turbo"  # Instead of gpt-4
)

# Reduce max tokens
config = ExpansionConfig(
    max_tokens=100  # Instead of 150
)
```

#### 2. Poor Expansion Quality

**Problem:** Expansions don't improve results

**Solutions:**
```python
# Add domain context
config = ExpansionConfig(
    domain_context="Your specific domain (e.g., 'medical research')"
)

# Lower temperature for consistency
config = ExpansionConfig(
    temperature=0.2  # Instead of 0.7
)

# Try different strategy
config = ExpansionConfig(
    strategy=ExpansionStrategy.HYDE  # Instead of KEYWORD
)
```

#### 3. High LLM Costs

**Problem:** Query expansion costs too much

**Solutions:**
```python
# Enable aggressive caching
config = ExpansionConfig(
    enable_cache=True,
    cache_ttl=7200  # 2 hours
)

# Use cheaper model
config = ExpansionConfig(
    llm_model="gpt-3.5-turbo"
)

# Reduce token usage
config = ExpansionConfig(
    max_tokens=50,
    max_additional_terms=3
)

# Only expand when needed
result = expander.expand(query, enable_expansion=should_expand(query))
```

#### 4. Cache Not Working

**Problem:** Every request hits LLM

**Solutions:**
```python
# Verify cache is enabled
config = ExpansionConfig(
    enable_cache=True  # Make sure this is True
)

# Check query normalization
query = query.strip().lower()  # Normalize before expanding

# Monitor cache stats
stats = expander.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

---

## Performance Optimization Tips

1. **Enable Caching:** Always enable caching in production
2. **Use Fast Models:** gpt-3.5-turbo is usually sufficient
3. **Batch Similar Queries:** Group similar queries to leverage cache
4. **Monitor Metrics:** Track performance and adjust configuration
5. **A/B Test:** Validate expansion improves results before full rollout
6. **Set Timeouts:** Configure appropriate timeout values
7. **Async Operations:** Use async code for parallel processing

---

## Next Steps

1. **Integrate into your RAG pipeline** using examples above
2. **Run A/B tests** to measure impact
3. **Monitor metrics** and optimize configuration
4. **Try different strategies** for different query types
5. **Customize prompts** for your domain

For more information, see:
- <!-- BROKEN LINK: Query Expansion README <!-- (broken link to: ../../rag_factory/strategies/query_expansion/README.md) --> --> Query Expansion README
- <!-- BROKEN LINK: Example Script <!-- (broken link to: ../../examples/query_expansion_example.py) --> --> Example Script
- <!-- BROKEN LINK: Unit Tests <!-- (broken link to: ../../tests/unit/strategies/query_expansion/) --> --> Unit Tests
