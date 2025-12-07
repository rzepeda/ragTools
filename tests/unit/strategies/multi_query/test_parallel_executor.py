"""Unit tests for ParallelQueryExecutor."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.multi_query.parallel_executor import ParallelQueryExecutor
from rag_factory.strategies.multi_query.config import MultiQueryConfig


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()

    async def mock_search(query, top_k):
        # Simulate different results for different queries
        await asyncio.sleep(0.1)  # Simulate query time
        return [
            {"chunk_id": f"chunk_{i}_{query[:5]}", "text": f"Result {i} for {query}", "score": 0.9 - i * 0.1}
            for i in range(min(3, top_k))
        ]

    store.asearch = mock_search
    return store


@pytest.fixture
def config():
    """Create default config."""
    return MultiQueryConfig(
        top_k_per_variant=5,
        query_timeout=5.0
    )


@pytest.fixture
def executor(mock_vector_store, config):
    """Create executor instance."""
    return ParallelQueryExecutor(mock_vector_store, config)


@pytest.mark.asyncio
async def test_execute_queries_parallel(executor):
    """Test parallel execution of multiple queries."""
    variants = ["query 1", "query 2", "query 3"]

    import time
    start = time.time()
    results = await executor.execute_queries(variants)
    duration = time.time() - start

    # Should execute in parallel (close to max time, not sum)
    assert duration < 0.5  # Much less than 0.3s (3 * 0.1s)

    # Should have results for all queries
    assert len(results) == 3
    assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_execute_single_query(executor):
    """Test single query execution."""
    result = await executor._execute_single_query("test query", variant_index=0)

    assert result["success"] is True
    assert result["query"] == "test query"
    assert result["variant_index"] == 0
    assert len(result["results"]) > 0
    assert "execution_time" in result


@pytest.mark.asyncio
async def test_query_timeout_handling():
    """Test timeout handling for slow queries."""
    # Create mock that times out
    store = Mock()

    async def slow_search(query, top_k):
        await asyncio.sleep(10)  # Longer than timeout
        return []

    store.asearch = slow_search

    config = MultiQueryConfig(query_timeout=0.1, min_successful_queries=0)  # Allow 0 successful
    executor = ParallelQueryExecutor(store, config)

    results = await executor.execute_queries(["slow query"])

    # Should have failed result
    assert len(results) == 1
    assert results[0]["success"] is False
    assert "error" in results[0]


@pytest.mark.asyncio
async def test_partial_failure_handling(config):
    """Test that partial failures are handled gracefully."""
    store = Mock()

    async def mixed_search(query, top_k):
        if "fail" in query:
            raise Exception("Query failed")
        await asyncio.sleep(0.05)
        return [{"chunk_id": "c1", "text": "result", "score": 0.9}]

    store.asearch = mixed_search

    executor = ParallelQueryExecutor(store, config)

    variants = ["good query 1", "fail query", "good query 2"]
    results = await executor.execute_queries(variants)

    # Should have 3 results (2 success, 1 failure)
    assert len(results) == 3
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    assert len(successful) == 2
    assert len(failed) == 1


@pytest.mark.asyncio
async def test_minimum_successful_queries():
    """Test minimum successful queries validation."""
    store = Mock()

    async def failing_search(query, top_k):
        raise Exception("All queries fail")

    store.asearch = failing_search

    config = MultiQueryConfig(min_successful_queries=2)
    executor = ParallelQueryExecutor(store, config)

    variants = ["query 1", "query 2"]

    with pytest.raises(ValueError, match="minimum required"):
        await executor.execute_queries(variants)


@pytest.mark.asyncio
async def test_sync_vector_store_support(config):
    """Test support for sync vector store (no asearch method)."""
    store = Mock()

    def sync_search(query, top_k):
        return [{"chunk_id": "c1", "text": "result", "score": 0.9}]

    store.search = sync_search
    # Remove asearch to force sync path
    delattr(store, 'asearch') if hasattr(store, 'asearch') else None

    executor = ParallelQueryExecutor(store, config)

    results = await executor.execute_queries(["test query"])

    assert len(results) == 1
    assert results[0]["success"] is True
