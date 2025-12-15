# Strategies Overview

RAG Factory provides 10 production-ready RAG strategies, each designed for specific use cases and retrieval patterns.

---

## All Strategies

| Strategy | Complexity | Speed | Accuracy | Best For |
|----------|------------|-------|----------|----------|
| <!-- BROKEN LINK: Agentic RAG <!-- (broken link to: agentic.md) --> --> Agentic RAG | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Complex multi-step queries |
| <!-- BROKEN LINK: Context-Aware Chunking <!-- (broken link to: chunking.md) --> --> Context-Aware Chunking | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Basic document splitting |
| [Contextual Retrieval](contextual.md) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | General-purpose retrieval |
| <!-- BROKEN LINK: Hierarchical RAG <!-- (broken link to: hierarchical.md) --> --> Hierarchical RAG | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Structured documents |
| <!-- BROKEN LINK: Knowledge Graph <!-- (broken link to: knowledge-graph.md) --> --> Knowledge Graph | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Entity relationships |
| <!-- BROKEN LINK: Late Chunking <!-- (broken link to: late-chunking.md) --> --> Late Chunking | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Full context preservation |
| <!-- BROKEN LINK: Multi-Query <!-- (broken link to: multi-query.md) --> --> Multi-Query | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Ambiguous queries |
| <!-- BROKEN LINK: Query Expansion <!-- (broken link to: query-expansion.md) --> --> Query Expansion | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Improving recall |
| <!-- BROKEN LINK: Reranking <!-- (broken link to: reranking.md) --> --> Reranking | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Improving precision |
| <!-- BROKEN LINK: Self-Reflective <!-- (broken link to: self-reflective.md) --> --> Self-Reflective | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Iterative refinement |

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
- <!-- BROKEN LINK: Individual Strategy Documentation <!-- (broken link to: agentic.md) --> --> Individual Strategy Documentation - Detailed docs for each strategy
- <!-- BROKEN LINK: Pipeline Tutorial <!-- (broken link to: ../tutorials/pipeline-setup.md) --> --> Pipeline Tutorial - Build strategy pipelines

---

## See Also

- [Architecture Overview](../architecture/overview.md)
- <!-- BROKEN LINK: Configuration Reference <!-- (broken link to: ../guides/configuration-reference.md) --> --> Configuration Reference
- <!-- BROKEN LINK: Performance Tuning <!-- (broken link to: ../guides/performance-tuning.md) --> --> Performance Tuning
