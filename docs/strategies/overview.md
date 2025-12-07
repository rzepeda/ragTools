# Strategies Overview

RAG Factory provides 10 production-ready RAG strategies, each designed for specific use cases and retrieval patterns.

---

## All Strategies

| Strategy | Complexity | Speed | Accuracy | Best For |
|----------|------------|-------|----------|----------|
| [Agentic RAG](agentic.md) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Complex multi-step queries |
| [Context-Aware Chunking](chunking.md) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Basic document splitting |
| [Contextual Retrieval](contextual.md) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | General-purpose retrieval |
| [Hierarchical RAG](hierarchical.md) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Structured documents |
| [Knowledge Graph](knowledge-graph.md) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Entity relationships |
| [Late Chunking](late-chunking.md) | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Full context preservation |
| [Multi-Query](multi-query.md) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Ambiguous queries |
| [Query Expansion](query-expansion.md) | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Improving recall |
| [Reranking](reranking.md) | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Improving precision |
| [Self-Reflective](self-reflective.md) | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Iterative refinement |

---

## Quick Selection Guide

### Need Maximum Accuracy?
→ Use **Reranking** or **Agentic RAG**

### Need Fast Performance?
→ Use **Context-Aware Chunking** or **Query Expansion**

### Have Structured Documents?
→ Use **Hierarchical RAG**

### Need Entity Relationships?
→ Use **Knowledge Graph**

### Have Ambiguous Queries?
→ Use **Multi-Query**

### Need Iterative Refinement?
→ Use **Self-Reflective**

---

## Strategy Categories

### Chunking Strategies
- **Context-Aware Chunking**: Intelligent document splitting
- **Hierarchical RAG**: Multi-level hierarchy preservation
- **Late Chunking**: Embed before chunking

### Query Enhancement Strategies
- **Multi-Query**: Generate query variants
- **Query Expansion**: Expand with related terms
- **Self-Reflective**: Iterative query refinement

### Result Enhancement Strategies
- **Contextual Retrieval**: Add context to chunks
- **Reranking**: Re-score with cross-encoder

### Advanced Strategies
- **Agentic RAG**: LLM agent-based tool selection
- **Knowledge Graph**: Graph-based knowledge extraction

---

## Combining Strategies

Strategies can be combined in pipelines for enhanced performance:

```python
from rag_factory.pipeline import StrategyPipeline
from rag_factory.factory import RAGFactory

factory = RAGFactory()

# High-accuracy pipeline
pipeline = StrategyPipeline([
    factory.create_strategy("multi_query", config),
    factory.create_strategy("contextual", config),
    factory.create_strategy("reranking", config)
])
```

---

## Next Steps

- [Strategy Selection Guide](../guides/strategy-selection.md) - Detailed selection criteria
- [Individual Strategy Documentation](agentic.md) - Detailed docs for each strategy
- [Pipeline Tutorial](../tutorials/pipeline-setup.md) - Build strategy pipelines

---

## See Also

- [Architecture Overview](../architecture/overview.md)
- [Configuration Reference](../guides/configuration-reference.md)
- [Performance Tuning](../guides/performance-tuning.md)
