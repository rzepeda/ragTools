# Epic 6: Multi-Query & Contextual Strategies - Story Documentation

This directory contains detailed story documentation for Epic 6, which focuses on implementing advanced RAG strategies that enhance retrieval through multiple perspectives and enriched context.

---

## Epic Overview

**Epic Goal:** Implement strategies that enhance retrieval through multiple perspectives and enriched context.

**Epic Story Points Total:** 26

**Dependencies:** Epic 4 (basic retrieval working)

---

## Stories

### Story 6.1: Implement Multi-Query RAG Strategy (13 points)
**File:** `story-6.1-multi-query-rag-strategy.md`

**Goal:** Generate multiple query variants and merge results for broader coverage.

**Key Features:**
- LLM-based query variant generation (3-5 variants)
- Parallel query execution using asyncio
- Result deduplication across variants
- Multiple ranking strategies (Max Score, RRF, Frequency Boost)
- Cost tracking and performance optimization

**Technical Highlights:**
- Async/parallel execution for performance
- Reciprocal Rank Fusion (RRF) ranking
- Configurable variant generation strategies
- Fallback to original query on failures

**Performance Targets:**
- Variant generation: <1s
- Parallel execution: <2s for 5 variants
- Total latency: <3s end-to-end

---

### Story 6.2: Implement Contextual Retrieval Strategy (13 points)
**File:** `story-6.2-contextual-retrieval-strategy.md`

**Goal:** Enrich chunks with document context before embedding for improved retrieval relevance.

**Key Features:**
- LLM-generated contextual descriptions for chunks
- Context prepending to chunks before embedding
- Dual storage (original + contextualized text)
- Batch processing for efficiency (10-50 chunks per batch)
- Comprehensive cost tracking
- Selective contextualization

**Technical Highlights:**
- Batch processing with parallel execution
- Context generation from multiple sources (metadata, hierarchy, surrounding chunks)
- Cost optimization and budget management
- Quality validation and A/B testing support

**Performance Targets:**
- Batch processing: <2s per batch (10-50 chunks)
- Throughput: >100 chunks/minute
- Cost: <$0.01 per 1000 chunks (GPT-3.5)

---

## Epic Architecture

### Overall Strategy Flow

```
User Query
    │
    ├─ Story 6.1: Multi-Query Strategy
    │   ├─ Generate query variants (LLM)
    │   ├─ Execute variants in parallel
    │   ├─ Deduplicate results
    │   └─ Rank and merge (RRF/Frequency Boost)
    │
    └─ Story 6.2: Contextual Retrieval Strategy
        ├─ Document ingestion with context generation
        ├─ Chunk enrichment (context prepending)
        ├─ Dual storage (original + contextualized)
        └─ Retrieve using contextualized embeddings
```

### Technology Stack

**Multi-Query Strategy:**
- asyncio for parallel execution
- LLM service for query generation
- Deduplication and ranking algorithms
- Reciprocal Rank Fusion (RRF)

**Contextual Strategy:**
- LLM service for context generation
- Batch processing utilities
- Cost tracking and monitoring
- Dual storage system

---

## Dependencies

### Required Services
- **LLM Service** (Epic 3): For query variant generation and context generation
- **Embedding Service** (Epic 3): For vectorizing contextualized chunks
- **Vector Store Service** (Epic 4): For semantic search
- **Database Service**: For dual storage of chunks

### Python Packages
```bash
# Multi-Query Strategy
pip install aiohttp>=3.8.0 tenacity>=8.0.0

# Contextual Strategy
# No additional dependencies (uses existing services)
```

---

## Configuration

### Multi-Query Configuration
```yaml
strategies:
  multi_query:
    enabled: true
    num_variants: 3
    ranking_strategy: "rrf"  # max_score, rrf, frequency_boost, hybrid
    query_timeout: 10.0
    final_top_k: 5
```

### Contextual Configuration
```yaml
strategies:
  contextual:
    enabled: true
    enable_contextualization: true
    batch_size: 20
    context_length_max: 200
    llm_model: "gpt-3.5-turbo"
    enable_cost_tracking: true
```

---

## Testing Strategy

### Unit Tests
Each story includes comprehensive unit tests with >90% code coverage:
- **Multi-Query**: Variant generator, parallel executor, deduplicator, ranker tests
- **Contextual**: Context generator, batch processor, cost tracker, storage tests

### Integration Tests
End-to-end tests with real services:
- Complete multi-query workflow
- Complete contextual retrieval workflow
- Quality comparison vs baseline
- Performance benchmarks

### Performance Benchmarks
- Multi-query latency: <3s end-to-end
- Contextual throughput: >100 chunks/minute
- Cost tracking accuracy verification

---

## Success Criteria

### Multi-Query Strategy (Story 6.1)
- [ ] Generates diverse query variants (avg cosine distance > 0.1)
- [ ] Parallel execution working efficiently
- [ ] Multiple ranking strategies implemented and tested
- [ ] Improved recall vs single-query baseline (>10% improvement)
- [ ] Performance targets met (<3s latency)

### Contextual Strategy (Story 6.2)
- [ ] Context generation produces relevant, accurate contexts
- [ ] Dual storage working correctly
- [ ] Batch processing meets performance targets (>100 chunks/min)
- [ ] Cost tracking accurate and comprehensive
- [ ] Improved retrieval accuracy vs baseline (>5% improvement)

### Overall Epic Success
- [ ] Both strategies integrated into RAG Factory
- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] Performance benchmarks meet requirements
- [ ] Quality improvements validated through A/B testing
- [ ] Documentation complete
- [ ] Code reviewed and merged

---

## Development Workflow

### Phase 1: Multi-Query Strategy (Sprint 7, Week 1)
1. Implement query variant generator
2. Implement parallel executor
3. Implement deduplicator and ranker
4. Write unit tests
5. Write integration tests
6. Performance optimization

### Phase 2: Contextual Strategy (Sprint 7, Week 2)
1. Implement context generator
2. Implement batch processor
3. Implement cost tracker and storage
4. Write unit tests
5. Write integration tests
6. Cost optimization

### Phase 3: Integration & Testing (Sprint 7, Week 3)
1. Integrate both strategies into main system
2. End-to-end testing
3. A/B testing for quality validation
4. Performance tuning
5. Documentation
6. Code review

---

## Quality Metrics

### Multi-Query Strategy
- **Recall Improvement**: >10% vs single-query baseline
- **Precision**: Maintain within 5% of baseline
- **Latency**: <3s for 5 variants
- **Variant Diversity**: Avg cosine distance > 0.1

### Contextual Strategy
- **Retrieval Accuracy**: >5% improvement vs non-contextualized
- **Throughput**: >100 chunks/minute
- **Cost Efficiency**: <$0.01 per 1000 chunks
- **Context Quality**: Relevant and concise (50-200 tokens)

---

## Common Patterns and Best Practices

### Async Programming
Both strategies use async/await for performance:
```python
async def example():
    # Always use async methods for best performance
    results = await strategy.aretrieve(query)

    # Use asyncio.gather for parallel operations
    results = await asyncio.gather(*tasks)

```

### Error Handling
Both strategies implement comprehensive error handling:
```python
async def example():
    # Fallback on failures
    if self.config.fallback_to_original:
        return await self._fallback_retrieve(query)

```

### Cost Management
Track and optimize LLM API costs:
```python
# Monitor costs
cost_summary = strategy.get_cost_summary()
print(f"Total cost: ${cost_summary['total_cost']:.4f}")

# Set budget limits
config = ContextualRetrievalConfig(
    max_cost_per_document=1.0,
    budget_alert_threshold=10.0
)
```

### Configuration Management
All parameters are configurable:
```python
# Use Pydantic models for type safety
config = MultiQueryConfig(
    num_variants=5,
    ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
    final_top_k=10
)
```

---

## Troubleshooting

### Multi-Query Issues

**Problem:** Variants are too similar
- **Solution:** Adjust LLM temperature, tune prompt template, check variant diversity metrics

**Problem:** Parallel execution is slow
- **Solution:** Check vector store performance, reduce num_variants, optimize query timeout

**Problem:** Results have low relevance
- **Solution:** Try different ranking strategies (RRF, hybrid), adjust frequency_boost_weight

### Contextual Issues

**Problem:** Context generation is expensive
- **Solution:** Use selective contextualization, reduce batch size, use cheaper LLM model

**Problem:** Low throughput
- **Solution:** Enable parallel batches, increase batch size, optimize LLM prompt length

**Problem:** Generated contexts are not relevant
- **Solution:** Tune prompt template, include more contextual sources, adjust context_length

---

## Migration Guide

### Adding Multi-Query to Existing System
```python
# Replace existing retrieval with multi-query
from rag_factory.strategies.multi_query import MultiQueryRAGStrategy

strategy = MultiQueryRAGStrategy(
    vector_store_service=existing_vector_store,
    llm_service=existing_llm,
    config=MultiQueryConfig()
)

# Use as drop-in replacement
results = strategy.retrieve(query, top_k=5)
```

### Adding Contextualization to Existing Chunks
```python
async def example():
    # Reprocess existing chunks with contextualization
    strategy = ContextualRetrievalStrategy(...)

    # Batch reprocess
    for document in existing_documents:
        chunks = get_chunks_for_document(document.id)
        await strategy.aindex_document(
            document.text,
            document.id,
            chunks,
            document.metadata
        )

```

---

## Performance Optimization Tips

### Multi-Query
1. Reduce `num_variants` if latency is critical
2. Use `RankingStrategy.MAX_SCORE` for fastest ranking
3. Enable `near_duplicate_detection` only when needed
4. Adjust `top_k_per_variant` to balance recall vs latency

### Contextual
1. Use larger batch sizes (20-50) for better throughput
2. Enable parallel batch processing
3. Use selective contextualization to reduce costs
4. Cache document-level contexts when possible
5. Use GPT-3.5-turbo for cost efficiency (vs GPT-4)

---

## References

- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval)
- [Multi-Query RAG Techniques](https://arxiv.org/abs/2305.03010)

---

## Support

For questions or issues:
1. Review story documentation in this directory
2. Check VERIFICATION.md for testing procedures
3. Review code examples in story files
4. Consult integration tests for usage patterns
