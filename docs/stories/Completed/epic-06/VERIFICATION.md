# Epic 6 Verification Guide

This document provides comprehensive verification procedures for Epic 6 stories: Multi-Query RAG Strategy (Story 6.1) and Contextual Retrieval Strategy (Story 6.2).

---

## Table of Contents

1. [Pre-Verification Checklist](#pre-verification-checklist)
2. [Story 6.1: Multi-Query RAG Strategy Verification](#story-61-multi-query-rag-strategy-verification)
3. [Story 6.2: Contextual Retrieval Strategy Verification](#story-62-contextual-retrieval-strategy-verification)
4. [Integration Verification](#integration-verification)
5. [Performance Verification](#performance-verification)
6. [Quality Verification](#quality-verification)
7. [Production Readiness Checklist](#production-readiness-checklist)

---

## Pre-Verification Checklist

Before verifying Epic 6 stories, ensure the following dependencies are in place:

### Environment Setup
- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test database accessible
- [ ] Test vector store accessible
- [ ] LLM API keys configured (OpenAI, Anthropic, or local LLM)

### Service Dependencies
- [ ] Embedding Service (Epic 3) working
- [ ] LLM Service (Epic 3) working
- [ ] Vector Store Service (Epic 4) working
- [ ] Database Service configured

### Test Data
- [ ] Test documents prepared (markdown, PDF, plain text)
- [ ] Test queries prepared for evaluation
- [ ] Ground truth labels available (if applicable)

### Commands
```bash
# Install dependencies
pip install aiohttp tenacity

# Verify services
python -c "import asyncio; print('Asyncio available')"

# Run all tests
pytest tests/unit/strategies/multi_query/ -v
pytest tests/unit/strategies/contextual/ -v
pytest tests/integration/strategies/ -v
```

---

## Story 6.1: Multi-Query RAG Strategy Verification

### Unit Test Verification

#### V6.1.1: Query Variant Generation Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/multi_query/test_variant_generator.py -v
```

**Verify:**
- [ ] Basic variant generation working
- [ ] Configurable number of variants (2-10)
- [ ] Original query inclusion/exclusion
- [ ] Variant validation (deduplication, length)
- [ ] Fallback on LLM failure
- [ ] Variant diversity validation

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.multi_query import QueryVariantGenerator, MultiQueryConfig

    config = MultiQueryConfig(num_variants=5, include_original=True)
    generator = QueryVariantGenerator(llm_service, config)

    query = "What is machine learning?"
    variants = await generator.generate_variants(query)

    # Verify
    assert len(variants) == 5
    assert query in variants
    assert len(set(variants)) == len(variants)  # No duplicates
    print("Generated variants:", variants)

```

#### V6.1.2: Parallel Executor Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/multi_query/test_parallel_executor.py -v
```

**Verify:**
- [ ] Parallel execution working
- [ ] Execution time is parallel (not sequential)
- [ ] Timeout handling working
- [ ] Partial failure handling
- [ ] Minimum successful queries requirement

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.multi_query import ParallelQueryExecutor
    import time

    executor = ParallelQueryExecutor(vector_store, config)
    variants = ["query 1", "query 2", "query 3", "query 4", "query 5"]

    start = time.time()
    results = await executor.execute_queries(variants)
    duration = time.time() - start

    # Verify parallel execution (should be close to max, not sum)
    print(f"Executed {len(variants)} queries in {duration:.2f}s")
    assert duration < 1.0  # Should be much less than sequential
    assert len([r for r in results if r["success"]]) >= config.min_successful_queries

```

#### V6.1.3: Deduplicator Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/multi_query/test_deduplicator.py -v
```

**Verify:**
- [ ] Exact deduplication by chunk ID
- [ ] Frequency tracking accurate
- [ ] Max score retention correct
- [ ] Failed queries skipped
- [ ] Empty results handled

**Manual Verification:**
```python
from rag_factory.strategies.multi_query import ResultDeduplicator

deduplicator = ResultDeduplicator(config)

# Simulate results with duplicates
query_results = [
    {"variant_index": 0, "success": True, "results": [
        {"chunk_id": "chunk_1", "score": 0.9},
        {"chunk_id": "chunk_2", "score": 0.8}
    ]},
    {"variant_index": 1, "success": True, "results": [
        {"chunk_id": "chunk_1", "score": 0.85},  # Duplicate
        {"chunk_id": "chunk_3", "score": 0.7}
    ]}
]

deduplicated = deduplicator.deduplicate(query_results)

# Verify
assert len(deduplicated) == 3  # 3 unique chunks
chunk_1 = next(c for c in deduplicated if c["chunk_id"] == "chunk_1")
assert chunk_1["frequency"] == 2
assert chunk_1["max_score"] == 0.9
print("Deduplication stats:", len(deduplicated), "unique chunks")
```

#### V6.1.4: Ranker Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/multi_query/test_ranker.py -v
```

**Verify:**
- [ ] Max Score ranking working
- [ ] Frequency Boost ranking working
- [ ] Reciprocal Rank Fusion (RRF) working
- [ ] Hybrid ranking working
- [ ] Top-k selection correct

**Manual Verification:**
```python
from rag_factory.strategies.multi_query import ResultRanker, RankingStrategy

config = MultiQueryConfig(
    ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
    final_top_k=5
)
ranker = ResultRanker(config)

# Sample results
results = [
    {"chunk_id": "c1", "max_score": 0.9, "frequency": 3, "variant_indices": [0, 1, 2]},
    {"chunk_id": "c2", "max_score": 0.95, "frequency": 1, "variant_indices": [0]},
    {"chunk_id": "c3", "max_score": 0.85, "frequency": 2, "variant_indices": [1, 2]}
]

ranked = ranker.rank(results)

# Verify
assert len(ranked) <= 5
assert all("final_score" in r for r in ranked)
assert ranked == sorted(ranked, key=lambda x: x["final_score"], reverse=True)
print("Ranked results:", [(r["chunk_id"], r["final_score"]) for r in ranked])
```

### Integration Test Verification

#### V6.1.5: End-to-End Multi-Query Workflow

**Run Tests:**
```bash
pytest tests/integration/strategies/test_multi_query_integration.py -v
```

**Verify:**
- [ ] Complete workflow working
- [ ] Async and sync wrappers working
- [ ] Variant diversity acceptable
- [ ] Performance requirements met (<3s)
- [ ] Fallback on failure working
- [ ] Different ranking strategies produce results

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.multi_query import MultiQueryRAGStrategy, MultiQueryConfig
    import asyncio
    import time

    config = MultiQueryConfig(
        num_variants=5,
        ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
        final_top_k=10
    )

    strategy = MultiQueryRAGStrategy(
        vector_store_service=vector_store,
        llm_service=llm,
        embedding_service=embedding_service,
        config=config
    )

    # Test retrieval
    query = "What are the benefits of machine learning?"

    start = time.time()
    results = await strategy.aretrieve(query)
    duration = time.time() - start

    # Verify
    assert len(results) <= 10
    assert duration < 3.0  # Performance requirement
    assert all("final_score" in r for r in results)
    assert all("frequency" in r for r in results)

    print(f"\nMulti-query retrieval completed in {duration:.2f}s")
    print(f"Results: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. Score: {r['final_score']:.3f}, Frequency: {r['frequency']}")

```

### Acceptance Criteria Verification

#### AC6.1.1: Query Variant Generation
- [ ] LLM integration working
- [ ] Generates 3-5 variants (configurable)
- [ ] Variants are diverse (avg distance > 0.1)
- [ ] Variants maintain intent
- [ ] Prompt template customizable
- [ ] Validation prevents invalid variants
- [ ] Fallback working

#### AC6.1.2: Parallel Execution
- [ ] Async execution implemented
- [ ] Concurrent execution verified
- [ ] Timeout handling working
- [ ] Error handling allows partial failures
- [ ] Performance: Time ≈ max(individual), not sum

#### AC6.1.3: Result Deduplication
- [ ] Exact deduplication working
- [ ] Near-duplicate detection (optional)
- [ ] Highest score retained
- [ ] Frequency tracking accurate
- [ ] Stats logged

#### AC6.1.4: Result Ranking
- [ ] 3+ ranking strategies implemented
- [ ] Strategy selection via config
- [ ] Results properly ordered
- [ ] Top-k selection working

---

## Story 6.2: Contextual Retrieval Strategy Verification

### Unit Test Verification

#### V6.2.1: Context Generator Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/contextual/test_context_generator.py -v
```

**Verify:**
- [ ] Basic context generation working
- [ ] Document metadata used
- [ ] Section hierarchy included
- [ ] Short chunks skipped
- [ ] Code blocks skipped (if configured)
- [ ] Fallback on error working
- [ ] Context length validation

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.contextual import ContextGenerator, ContextualRetrievalConfig

    config = ContextualRetrievalConfig(
        context_length_min=50,
        context_length_max=200,
        skip_code_blocks=True
    )
    generator = ContextGenerator(llm_service, config)

    chunk = {
        "chunk_id": "chunk_1",
        "text": "Machine learning algorithms learn patterns from data to make predictions.",
        "metadata": {
            "document_id": "ml_guide",
            "section_hierarchy": ["Chapter 1", "Introduction"]
        }
    }

    document_context = {"title": "ML Guide for Beginners"}

    context = await generator.generate_context(chunk, document_context)

    # Verify
    assert context is not None
    assert 50 <= len(context.split()) <= 250  # Rough token count
    print("Generated context:", context)

```

#### V6.2.2: Batch Processor Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/contextual/test_batch_processor.py -v
```

**Verify:**
- [ ] Batch creation working
- [ ] Parallel batch processing working
- [ ] Cost tracking during batch
- [ ] Error handling in batch
- [ ] Progress tracking

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.contextual import BatchProcessor, CostTracker
    import time

    processor = BatchProcessor(context_generator, cost_tracker, config)

    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Content {i} " * 30, "metadata": {}}
        for i in range(50)
    ]

    start = time.time()
    processed = await processor.process_chunks(chunks)
    duration = time.time() - start

    chunks_per_minute = (len(processed) / duration) * 60

    # Verify
    assert len(processed) == 50
    assert chunks_per_minute >= 100  # Performance requirement
    print(f"Processed {len(processed)} chunks in {duration:.2f}s ({chunks_per_minute:.0f} chunks/min)")

```

#### V6.2.3: Cost Tracker Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/contextual/test_cost_tracker.py -v
```

**Verify:**
- [ ] Cost calculation correct
- [ ] Chunk cost recording accurate
- [ ] Cost summary generation working
- [ ] Budget alert triggers
- [ ] Reset working

**Manual Verification:**
```python
from rag_factory.strategies.contextual import CostTracker

tracker = CostTracker(config)

# Record costs
for i in range(100):
    tracker.record_chunk_cost(
        chunk_id=f"chunk_{i}",
        input_tokens=500,
        output_tokens=100,
        cost=0.001
    )

summary = tracker.get_summary()

# Verify
assert summary["total_chunks"] == 100
assert summary["total_cost"] == 0.1
assert summary["avg_cost_per_chunk"] == 0.001

print("Cost summary:", summary)
```

#### V6.2.4: Storage Manager Tests

**Run Tests:**
```bash
pytest tests/unit/strategies/contextual/test_storage.py -v
```

**Verify:**
- [ ] Dual storage working
- [ ] Original text stored
- [ ] Context stored separately
- [ ] Contextualized text stored
- [ ] Retrieval format options working

**Manual Verification:**
```python
from rag_factory.strategies.contextual import ContextualStorageManager

storage = ContextualStorageManager(database, config)

chunks = [
    {
        "chunk_id": "chunk_1",
        "text": "Original text",
        "context_description": "Generated context",
        "contextualized_text": "Context: Generated context\n\nOriginal text",
        "document_id": "doc_1"
    }
]

# Store
storage.store_chunks(chunks)

# Retrieve in different formats
original = storage.retrieve_chunks(["chunk_1"], return_format="original")
contextualized = storage.retrieve_chunks(["chunk_1"], return_format="contextualized")
both = storage.retrieve_chunks(["chunk_1"], return_format="both")

# Verify
assert original[0]["text"] == "Original text"
assert "context" in contextualized[0]["text"] or "Context" in contextualized[0]["text"]
assert "original_text" in both[0] and "contextualized_text" in both[0]
```

### Integration Test Verification

#### V6.2.5: End-to-End Contextual Retrieval

**Run Tests:**
```bash
pytest tests/integration/strategies/test_contextual_integration.py -v
```

**Verify:**
- [ ] Complete workflow working
- [ ] Context generation producing relevant contexts
- [ ] Dual storage working
- [ ] Cost tracking accurate
- [ ] Retrieval quality improved

**Manual Verification:**
```python
async def example():
    from rag_factory.strategies.contextual import ContextualRetrievalStrategy
    import asyncio

    strategy = ContextualRetrievalStrategy(
        vector_store_service=vector_store,
        database_service=database,
        llm_service=llm,
        embedding_service=embedding_service,
        config=config
    )

    # Prepare chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "text": f"Machine learning content about topic {i}...",
            "metadata": {"section_hierarchy": [f"Section {i}"]}
        }
        for i in range(20)
    ]

    # Index with contextualization
    result = await strategy.aindex_document(
        document="Full document text",
        document_id="doc_1",
        chunks=chunks,
        document_metadata={"title": "ML Tutorial"}
    )

    # Verify indexing
    assert result["total_chunks"] == 20
    assert result["contextualized_chunks"] > 0
    assert result["total_cost"] > 0

    print(f"Indexed {result['total_chunks']} chunks")
    print(f"Contextualized: {result['contextualized_chunks']}")
    print(f"Cost: ${result['total_cost']:.4f}")

    # Retrieve
    results = strategy.retrieve("machine learning concepts", top_k=5)

    # Verify retrieval
    assert len(results) <= 5
    assert all("text" in r for r in results)

    print(f"\nRetrieved {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['chunk_id']}: {r['text'][:100]}...")

```

### Acceptance Criteria Verification

#### AC6.2.1: Context Generation
- [ ] LLM integration working
- [ ] Context length configurable (50-200 tokens)
- [ ] Customizable prompts
- [ ] Context includes doc/section info
- [ ] Context accurate and relevant
- [ ] Batch processing working

#### AC6.2.2: Dual Storage
- [ ] Schema supports dual storage
- [ ] Original text stored
- [ ] Context stored
- [ ] Contextualized text stored
- [ ] Embeddings from contextualized
- [ ] Retrieval returns original (configurable)

#### AC6.2.3: Performance
- [ ] Batch processing <2s per batch
- [ ] Throughput >100 chunks/min
- [ ] Large documents <5min for 1000 chunks
- [ ] Async/parallel working

#### AC6.2.4: Cost Tracking
- [ ] Token usage tracked
- [ ] Cost calculated accurately
- [ ] Cost reporting available
- [ ] Budget limits enforceable

---

## Integration Verification

### V6.3: Combined Strategies Test

Verify that both strategies can work together:

```python
async def example():
    # Use contextual retrieval for indexing
    contextual_strategy = ContextualRetrievalStrategy(...)
    await contextual_strategy.aindex_document(document, doc_id, chunks)

    # Use multi-query for retrieval
    multi_query_strategy = MultiQueryRAGStrategy(...)
    results = await multi_query_strategy.aretrieve(query)

    # Verify
    assert len(results) > 0
    print("Combined strategies working:", len(results), "results")

```

### V6.4: Switching Between Strategies

Verify easy switching:

```python
async def example():
    # Create both strategies with same services
    contextual = ContextualRetrievalStrategy(vector_store, db, llm, embedding, contextual_config)
    multi_query = MultiQueryRAGStrategy(vector_store, llm, embedding, multi_query_config)

    # Use contextual for indexing
    await contextual.aindex_document(doc, doc_id, chunks)

    # Use either for retrieval
    results_mq = await multi_query.aretrieve(query)
    results_ctx = contextual.retrieve(query)

    # Verify both work
    assert len(results_mq) > 0
    assert len(results_ctx) > 0

```

---

## Performance Verification

### P6.1: Multi-Query Performance Benchmarks

**Run Benchmarks:**
```bash
pytest tests/benchmarks/test_multi_query_performance.py -v
```

**Verify Targets:**
- [ ] Variant generation: <1s
- [ ] Parallel execution: <2s for 5 variants
- [ ] End-to-end latency: <3s
- [ ] Throughput: >=3 queries/second

**Manual Benchmark:**
```python
async def example():
    import time
    import statistics

    latencies = []

    for i in range(20):
        query = f"Test query {i}"

        start = time.time()
        results = await strategy.aretrieve(query)
        duration = time.time() - start

        latencies.append(duration)

    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    qps = 20 / sum(latencies)

    print(f"Average latency: {avg_latency:.2f}s")
    print(f"P95 latency: {p95_latency:.2f}s")
    print(f"Throughput: {qps:.2f} queries/second")

    assert avg_latency < 3.0
    assert qps >= 3.0

```

### P6.2: Contextual Performance Benchmarks

**Run Benchmarks:**
```bash
pytest tests/benchmarks/test_contextual_performance.py -v
```

**Verify Targets:**
- [ ] Batch processing: <2s per batch (10-50 chunks)
- [ ] Throughput: >100 chunks/minute
- [ ] Large document: <5min for 1000 chunks

**Manual Benchmark:**
```python
async def example():
    import time

    # Generate large chunk set
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Content {i} " * 50, "metadata": {}}
        for i in range(500)
    ]

    start = time.time()
    result = await strategy.aindex_document("doc", "large_doc", chunks)
    duration = time.time() - start

    chunks_per_minute = (len(chunks) / duration) * 60
    time_per_1000 = (duration / len(chunks)) * 1000

    print(f"Processed {len(chunks)} chunks in {duration:.2f}s")
    print(f"Throughput: {chunks_per_minute:.0f} chunks/minute")
    print(f"Estimated time for 1000 chunks: {time_per_1000:.2f}s")

    assert chunks_per_minute >= 100
    assert time_per_1000 < 300  # 5 minutes

```

---

## Quality Verification

### Q6.1: Multi-Query Quality Metrics

**Verify Recall Improvement:**

```python
async def example():
    # Baseline: Single query
    baseline_results = vector_store.search(query, top_k=20)

    # Multi-query
    multi_query_results = await multi_query_strategy.aretrieve(query)

    # Calculate recall improvement
    baseline_ids = {r["chunk_id"] for r in baseline_results}
    multi_query_ids = {r["chunk_id"] for r in multi_query_results}

    additional_relevant = multi_query_ids - baseline_ids
    recall_improvement = len(additional_relevant) / len(baseline_ids)

    print(f"Recall improvement: {recall_improvement * 100:.1f}%")
    assert recall_improvement >= 0.10  # 10% improvement target

```

**Verify Variant Diversity:**

```python
async def example():
    # Generate variants
    variants = await variant_generator.generate_variants(query)

    # Calculate diversity using embeddings
    if len(variants) > 1:
        embed_result = embedding_service.embed(variants)
        embeddings = embed_result.embeddings

        # Calculate pairwise cosine distances
        import numpy as np
        distances = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                distance = 1 - sim
                distances.append(distance)

        avg_distance = np.mean(distances)

        print(f"Average variant distance: {avg_distance:.3f}")
        assert avg_distance >= 0.1  # Variants should be diverse

```

### Q6.2: Contextual Quality Metrics

**Verify Retrieval Accuracy Improvement:**

```python
async def example():
    # Test with and without contextualization
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain supervised learning"
    ]

    # Without contextualization
    config_without = ContextualRetrievalConfig(enable_contextualization=False)
    strategy_without = ContextualRetrievalStrategy(..., config=config_without)

    # With contextualization
    config_with = ContextualRetrievalConfig(enable_contextualization=True)
    strategy_with = ContextualRetrievalStrategy(..., config=config_with)

    # Index and retrieve
    await strategy_without.aindex_document(doc, "doc_wo", chunks)
    await strategy_with.aindex_document(doc, "doc_w", chunks)

    accuracy_improvements = []

    for query in test_queries:
        results_without = strategy_without.retrieve(query, top_k=10)
        results_with = strategy_with.retrieve(query, top_k=10)

        # Compare top result scores (proxy for relevance)
        if results_without and results_with:
            score_without = results_without[0]["score"]
            score_with = results_with[0]["score"]
            improvement = (score_with - score_without) / score_without
            accuracy_improvements.append(improvement)

    avg_improvement = np.mean(accuracy_improvements)
    print(f"Average accuracy improvement: {avg_improvement * 100:.1f}%")
    assert avg_improvement >= 0.05  # 5% improvement target

```

**Verify Context Quality:**

```python
async def example():
    # Generate contexts for sample chunks
    sample_chunks = chunks[:10]

    for chunk in sample_chunks:
        context = await context_generator.generate_context(chunk)

        if context:
            # Check context properties
            token_count = context_generator._count_tokens(context)

            # Verify length
            assert 50 <= token_count <= 200

            # Verify relevance (simple check - context should mention key terms)
            chunk_text = chunk["text"].lower()
            context_lower = context.lower()

            # Extract key nouns from chunk (simplified)
            chunk_words = set(chunk_text.split())
            context_words = set(context_lower.split())

            overlap = len(chunk_words & context_words) / len(chunk_words)

            print(f"Chunk: {chunk['chunk_id']}")
            print(f"Context length: {token_count} tokens")
            print(f"Word overlap: {overlap * 100:.1f}%")
            print(f"Context: {context}")
            print()

            # Context should have some overlap with chunk content
            assert overlap > 0.1

```

---

## Production Readiness Checklist

### Code Quality
- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] No linting errors (`pylint`, `flake8`)
- [ ] Type hints present (`mypy` passing)
- [ ] Code reviewed and approved

### Performance
- [ ] Multi-query: <3s end-to-end latency
- [ ] Contextual: >100 chunks/minute throughput
- [ ] Memory usage acceptable (<500MB overhead)
- [ ] No memory leaks detected
- [ ] Performance benchmarks documented

### Quality
- [ ] Multi-query recall improvement >10% validated
- [ ] Contextual accuracy improvement >5% validated
- [ ] Variant diversity metrics acceptable
- [ ] Context quality validated
- [ ] A/B test results positive

### Configuration
- [ ] All parameters configurable via config files
- [ ] Default configurations sensible
- [ ] Environment-specific configs (dev, staging, prod)
- [ ] Configuration documentation complete

### Monitoring & Observability
- [ ] Comprehensive logging implemented
- [ ] Metrics tracked (latency, cost, quality)
- [ ] Error tracking and alerting
- [ ] Cost monitoring and alerts
- [ ] Performance monitoring dashboards

### Documentation
- [ ] Story documentation complete
- [ ] API documentation complete
- [ ] Configuration guide complete
- [ ] Deployment guide complete
- [ ] Troubleshooting guide complete
- [ ] Code examples provided

### Security
- [ ] API keys securely managed
- [ ] No secrets in code or configs
- [ ] Input validation implemented
- [ ] SQL injection prevention (if applicable)
- [ ] Rate limiting considered

### Cost Management
- [ ] Cost tracking implemented
- [ ] Budget alerts configured
- [ ] Cost optimization strategies documented
- [ ] Cost projections calculated

### Deployment
- [ ] Database migrations tested
- [ ] Rollback plan documented
- [ ] Deployment scripts tested
- [ ] Health checks implemented
- [ ] Gradual rollout plan prepared

---

## Verification Sign-off

### Story 6.1: Multi-Query RAG Strategy
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Quality metrics validated
- [ ] Code reviewed
- [ ] Documentation complete

**Verified by:** __________________ **Date:** __________

### Story 6.2: Contextual Retrieval Strategy
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Quality metrics validated
- [ ] Cost tracking verified
- [ ] Code reviewed
- [ ] Documentation complete

**Verified by:** __________________ **Date:** __________

### Epic 6: Complete Integration
- [ ] Both strategies integrated
- [ ] Combined testing complete
- [ ] Production readiness verified
- [ ] Deployment approved

**Verified by:** __________________ **Date:** __________

---

## Troubleshooting Common Issues

### Multi-Query Issues

**Issue:** Variants are not diverse enough
- Check LLM temperature (should be 0.7-0.9)
- Review prompt template
- Verify variant validation logic
- Check embedding model for diversity calculation

**Issue:** Parallel execution is slow
- Verify asyncio event loop configuration
- Check vector store performance
- Reduce number of variants
- Check network latency to services

**Issue:** Ranking produces poor results
- Try different ranking strategies
- Adjust frequency_boost_weight
- Check RRF k parameter
- Validate deduplication is working

### Contextual Issues

**Issue:** Context generation is too expensive
- Enable selective contextualization
- Reduce batch size to increase parallelism
- Use cheaper LLM model (GPT-3.5 vs GPT-4)
- Optimize prompt length

**Issue:** Contexts are not relevant
- Review and tune prompt templates
- Include more contextual sources
- Check document metadata quality
- Validate LLM model performance

**Issue:** Low throughput
- Enable parallel batch processing
- Increase batch size
- Check LLM API rate limits
- Verify async execution working

---

## Appendix: Verification Scripts

### Script 1: Complete Multi-Query Verification

```bash
#!/bin/bash

echo "=== Multi-Query RAG Strategy Verification ==="

echo "Running unit tests..."
pytest tests/unit/strategies/multi_query/ -v --cov

echo "Running integration tests..."
pytest tests/integration/strategies/test_multi_query_integration.py -v

echo "Running performance benchmarks..."
pytest tests/benchmarks/test_multi_query_performance.py -v

echo "Verification complete!"
```

### Script 2: Complete Contextual Verification

```bash
#!/bin/bash

echo "=== Contextual Retrieval Strategy Verification ==="

echo "Running unit tests..."
pytest tests/unit/strategies/contextual/ -v --cov

echo "Running integration tests..."
pytest tests/integration/strategies/test_contextual_integration.py -v

echo "Running performance benchmarks..."
pytest tests/benchmarks/test_contextual_performance.py -v

echo "Verification complete!"
```

### Script 3: Quality Metrics Verification

```python
#!/usr/bin/env python3
"""
Quality metrics verification script for Epic 6.
"""

import asyncio
from rag_factory.strategies.multi_query import MultiQueryRAGStrategy
from rag_factory.strategies.contextual import ContextualRetrievalStrategy

async def verify_quality():
    print("=== Epic 6 Quality Verification ===\n")

    # Multi-query quality
    print("1. Multi-Query Recall Improvement")
    # ... implementation ...

    # Variant diversity
    print("2. Variant Diversity")
    # ... implementation ...

    # Contextual accuracy
    print("3. Contextual Accuracy Improvement")
    # ... implementation ...

    # Context quality
    print("4. Context Quality")
    # ... implementation ...

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    asyncio.run(verify_quality())
```

---

## Support & Questions

For verification support:
1. Review story documentation in `story-6.1-*.md` and `story-6.2-*.md`
2. Check README.md for architecture overview
3. Review test files for usage examples
4. Contact development team for assistance
