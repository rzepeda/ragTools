"""
Unit tests for Strategy Pipeline implementation.

This module contains comprehensive unit tests for the StrategyPipeline class,
covering construction, execution modes, error handling, result merging,
and configuration loading.
"""

import time
from typing import Any, Dict, List

import pytest

from rag_factory.pipeline import (
    ExecutionMode,
    PipelineResult,
    PipelineStage,
    StrategyPipeline,
)
from rag_factory.strategies.base import Chunk, IRAGStrategy


# Test helper classes
class DummyStrategy(IRAGStrategy):
    """Dummy strategy for testing purposes."""

    def initialize(self, config: Any) -> None:
        """Initialize the dummy strategy."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
        """Prepare data (not used in these tests)."""
        return None

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve chunks."""
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve chunks."""
        return []

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query."""
        return ""


# TC3.1: Pipeline Construction Tests
def test_pipeline_can_be_created() -> None:
    """Test pipeline can be instantiated."""
    pipeline = StrategyPipeline()
    assert isinstance(pipeline, StrategyPipeline)
    assert len(pipeline.stages) == 0


def test_pipeline_add_stage() -> None:
    """Test adding stages to pipeline."""
    pipeline = StrategyPipeline()
    strategy = DummyStrategy()
    pipeline.add_stage(strategy, "test_stage")

    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].name == "test_stage"


def test_pipeline_chaining() -> None:
    """Test add_stage supports method chaining."""
    class Strategy1(IRAGStrategy):
        """Test strategy 1."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Test strategy 2."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = (StrategyPipeline()
                .add_stage(Strategy1(), "stage1")
                .add_stage(Strategy2(), "stage2"))

    assert len(pipeline.stages) == 2


# TC3.2: Sequential Execution Tests
def test_sequential_execution_order() -> None:
    """Test strategies execute in correct order."""
    execution_order: List[int] = []

    class Strategy1(IRAGStrategy):
        """Test strategy that tracks execution order."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve and track execution."""
            execution_order.append(1)
            return [Chunk("Result1", {}, 0.9, "doc1", "chunk1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Test strategy that tracks execution order."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve and track execution."""
            execution_order.append(2)
            return [Chunk("Result2", {}, 0.8, "doc2", "chunk2")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    pipeline.add_stage(Strategy1(), "strategy1")
    pipeline.add_stage(Strategy2(), "strategy2")

    result = pipeline.execute("test query", top_k=5)

    assert execution_order == [1, 2]
    assert len(result.final_results) == 2


def test_sequential_execution_collects_results() -> None:
    """Test sequential execution collects all results."""
    class Strategy1(IRAGStrategy):
        """Test strategy returning result A."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("A", {}, 0.9, "doc1", "chunk1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Test strategy returning result B."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("B", {}, 0.8, "doc2", "chunk2")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    pipeline.add_stage(Strategy1(), "s1")
    pipeline.add_stage(Strategy2(), "s2")

    result = pipeline.execute("query", top_k=5)

    assert len(result.final_results) == 2
    texts = [c.text for c in result.final_results]
    assert "A" in texts
    assert "B" in texts


# TC3.3: Parallel Execution Tests
@pytest.mark.asyncio
async def test_parallel_execution() -> None:
    """Test strategies execute in parallel."""
    import asyncio

    execution_times: Dict[str, float] = {}

    class SlowStrategy(IRAGStrategy):
        """Slow strategy for parallel testing."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve with delay."""
            start = time.time()
            await asyncio.sleep(0.1)
            execution_times["slow"] = time.time() - start
            return [Chunk("Slow", {}, 0.9, "doc1", "chunk1")]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FastStrategy(IRAGStrategy):
        """Fast strategy for parallel testing."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve with delay."""
            start = time.time()
            await asyncio.sleep(0.05)
            execution_times["fast"] = time.time() - start
            return [Chunk("Fast", {}, 0.8, "doc2", "chunk2")]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    pipeline.add_stage(SlowStrategy(), "slow")
    pipeline.add_stage(FastStrategy(), "fast")

    start_time = time.time()
    result = await pipeline.aexecute("query", top_k=5)
    total_time = time.time() - start_time

    # Parallel execution should be faster than sequential
    assert total_time < 0.15  # Less than sum of individual times
    assert len(result.final_results) == 2


# TC3.4: Error Handling Tests
def test_strategy_error_caught() -> None:
    """Test pipeline catches strategy errors."""
    class FailingStrategy(IRAGStrategy):
        """Strategy that always fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Strategy failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class WorkingStrategy(IRAGStrategy):
        """Strategy that works correctly."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("Success", {}, 0.9, "doc1", "chunk1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    stage = PipelineStage(
        strategy=FailingStrategy(),
        name="failing",
        required=False
    )
    pipeline.stages.append(stage)
    pipeline.add_stage(WorkingStrategy(), "working")

    result = pipeline.execute("query", top_k=5)

    assert len(result.errors) == 1
    assert result.errors[0]["stage"] == "failing"
    assert len(result.final_results) == 1


def test_fallback_strategy_executed() -> None:
    """Test fallback strategy used on primary failure."""
    class PrimaryStrategy(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Primary failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FallbackStrategy(IRAGStrategy):
        """Fallback strategy that succeeds."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("Fallback", {}, 0.7, "doc1", "chunk1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    pipeline.add_stage(
        PrimaryStrategy(),
        "primary",
        fallback=FallbackStrategy()
    )

    result = pipeline.execute("query", top_k=5)

    assert len(result.final_results) == 1
    assert result.final_results[0].text == "Fallback"


# TC3.5: Result Merging Tests
def test_duplicate_results_removed() -> None:
    """Test duplicate results are deduplicated."""
    class Strategy1(IRAGStrategy):
        """Strategy returning duplicate and unique results."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [
                Chunk("Duplicate", {}, 0.9, "doc1", "chunk1"),
                Chunk("Unique1", {}, 0.8, "doc2", "chunk2")
            ]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Strategy returning duplicate and unique results."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [
                Chunk("Duplicate", {}, 0.85, "doc1", "chunk1"),
                Chunk("Unique2", {}, 0.75, "doc3", "chunk3")
            ]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    pipeline.add_stage(Strategy1(), "s1")
    pipeline.add_stage(Strategy2(), "s2")

    result = pipeline.execute("query", top_k=5)

    texts = [c.text for c in result.final_results]
    assert texts.count("Duplicate") == 1
    assert len(result.final_results) == 3


def test_results_sorted_by_score() -> None:
    """Test merged results sorted by relevance score."""
    class Strategy1(IRAGStrategy):
        """Strategy returning low-score result."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("Low", {}, 0.6, "doc1", "chunk1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Strategy returning high-score result."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("High", {}, 0.95, "doc2", "chunk2")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    pipeline.add_stage(Strategy1(), "s1")
    pipeline.add_stage(Strategy2(), "s2")

    result = pipeline.execute("query", top_k=5)

    assert result.final_results[0].text == "High"
    assert result.final_results[0].score == 0.95


# TC3.6: Performance Tracking Tests
def test_performance_metrics_collected() -> None:
    """Test execution time tracked for each stage."""
    class Strategy1(IRAGStrategy):
        """Strategy with artificial delay."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve with delay."""
            time.sleep(0.01)
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    pipeline.add_stage(Strategy1(), "s1")

    result = pipeline.execute("query", top_k=5)

    assert "s1" in result.performance_metrics
    assert result.performance_metrics["s1"] > 0
    assert "total" in result.performance_metrics


# TC3.7: Configuration Loading Tests
def test_from_config_creates_pipeline() -> None:
    """Test pipeline can be created from config dict."""
    from rag_factory.factory import RAGFactory

    config = {
        "mode": "sequential",
        "stages": [
            {
                "strategy": "dummy",
                "name": "stage1",
                "config": {"chunk_size": 512}
            }
        ]
    }

    # Register dummy strategy
    RAGFactory.register_strategy("dummy", DummyStrategy, override=True)

    pipeline = StrategyPipeline.from_config(config)

    assert len(pipeline.stages) == 1
    assert pipeline.mode == ExecutionMode.SEQUENTIAL


# Additional tests for 100% coverage
def test_cascade_mode_not_implemented() -> None:
    """Test CASCADE mode raises NotImplementedError."""
    pipeline = StrategyPipeline(mode=ExecutionMode.CASCADE)
    pipeline.add_stage(DummyStrategy(), "stage1")

    with pytest.raises(NotImplementedError, match="Mode.*not implemented"):
        pipeline.execute("query", top_k=5)


@pytest.mark.asyncio
async def test_cascade_mode_not_implemented_async() -> None:
    """Test CASCADE mode raises NotImplementedError in async."""
    pipeline = StrategyPipeline(mode=ExecutionMode.CASCADE)
    pipeline.add_stage(DummyStrategy(), "stage1")

    with pytest.raises(NotImplementedError, match="Mode.*not implemented"):
        await pipeline.aexecute("query", top_k=5)


def test_required_stage_failure_raises_exception() -> None:
    """Test required stage failure raises exception."""
    class FailingRequiredStrategy(IRAGStrategy):
        """Strategy that fails and is required."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise ValueError("Required strategy failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    # Add required failing stage (required=True is default)
    pipeline.add_stage(FailingRequiredStrategy(), "required_failing")

    with pytest.raises(ValueError, match="Required strategy failed"):
        pipeline.execute("query", top_k=5)


@pytest.mark.asyncio
async def test_required_stage_failure_raises_exception_async() -> None:
    """Test required stage failure raises exception in async mode."""
    class FailingRequiredAsyncStrategy(IRAGStrategy):
        """Strategy that fails in async mode and is required."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise ValueError("Required async strategy failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    pipeline.add_stage(FailingRequiredAsyncStrategy(), "required_failing")

    with pytest.raises(ValueError, match="Required async strategy failed"):
        await pipeline.aexecute("query", top_k=5)


def test_fallback_failure_when_required() -> None:
    """Test fallback failure on required stage raises exception."""
    class FailingPrimary(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Primary failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingFallback(IRAGStrategy):
        """Fallback strategy that also fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Fallback failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline()
    pipeline.add_stage(
        FailingPrimary(),
        "primary",
        fallback=FailingFallback()
    )

    with pytest.raises(RuntimeError):
        pipeline.execute("query", top_k=5)


@pytest.mark.asyncio
async def test_fallback_failure_when_required_async() -> None:
    """Test fallback failure on required stage raises exception in async."""
    class FailingPrimaryAsync(IRAGStrategy):
        """Primary async strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Primary async failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingFallbackAsync(IRAGStrategy):
        """Fallback async strategy that also fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Fallback async failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    pipeline.add_stage(
        FailingPrimaryAsync(),
        "primary",
        fallback=FailingFallbackAsync()
    )

    with pytest.raises(RuntimeError):
        await pipeline.aexecute("query", top_k=5)


@pytest.mark.asyncio
async def test_parallel_required_stage_failure() -> None:
    """Test parallel execution with required stage failure."""
    class FailingParallelRequired(IRAGStrategy):
        """Required strategy that fails in parallel mode."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Parallel required failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    pipeline.add_stage(FailingParallelRequired(), "failing_required")

    with pytest.raises(RuntimeError, match="Parallel required failed"):
        await pipeline.aexecute("query", top_k=5)


@pytest.mark.asyncio
async def test_parallel_required_fallback_failure() -> None:
    """Test parallel execution with required stage and failing fallback."""
    class FailingParallelPrimary(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Parallel primary failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingParallelFallback(IRAGStrategy):
        """Fallback strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Parallel fallback failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    pipeline.add_stage(
        FailingParallelPrimary(),
        "primary",
        fallback=FailingParallelFallback()
    )

    with pytest.raises(RuntimeError, match="Parallel primary failed"):
        await pipeline.aexecute("query", top_k=5)


def test_non_required_fallback_failure_continues() -> None:
    """Test non-required stage with failing fallback continues pipeline."""
    class FailingPrimaryNonReq(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Primary failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingFallbackNonReq(IRAGStrategy):
        """Fallback strategy that also fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Fallback failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    from rag_factory.pipeline import PipelineStage

    pipeline = StrategyPipeline()
    stage = PipelineStage(
        strategy=FailingPrimaryNonReq(),
        name="non_required",
        fallback=FailingFallbackNonReq(),
        required=False
    )
    pipeline.stages.append(stage)

    # Should not raise, just continue
    result = pipeline.execute("query", top_k=5)
    assert len(result.errors) == 2  # Both primary and fallback errors


@pytest.mark.asyncio
async def test_non_required_fallback_failure_continues_async() -> None:
    """Test non-required stage with failing fallback continues in async."""
    class FailingPrimaryAsyncNonReq(IRAGStrategy):
        """Primary async strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Primary async failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingFallbackAsyncNonReq(IRAGStrategy):
        """Fallback async strategy that also fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Fallback async failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    from rag_factory.pipeline import PipelineStage

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    stage = PipelineStage(
        strategy=FailingPrimaryAsyncNonReq(),
        name="non_required",
        fallback=FailingFallbackAsyncNonReq(),
        required=False
    )
    pipeline.stages.append(stage)

    # Should not raise, just continue
    result = await pipeline.aexecute("query", top_k=5)
    assert len(result.errors) == 2  # Both primary and fallback errors


@pytest.mark.asyncio
async def test_parallel_non_required_fallback_success() -> None:
    """Test parallel execution with non-required stage and successful fallback."""
    class FailingParallelPrimaryNonReq(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Parallel primary failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class WorkingParallelFallback(IRAGStrategy):
        """Fallback strategy that works."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - returns result."""
            return [Chunk("Fallback", {}, 0.7, "doc1", "c1")]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    from rag_factory.pipeline import PipelineStage

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    stage = PipelineStage(
        strategy=FailingParallelPrimaryNonReq(),
        name="non_required_with_fallback",
        fallback=WorkingParallelFallback(),
        required=False
    )
    pipeline.stages.append(stage)

    result = await pipeline.aexecute("query", top_k=5)
    # Should have result from fallback
    assert len(result.final_results) == 1
    assert result.final_results[0].text == "Fallback"


@pytest.mark.asyncio
async def test_parallel_non_required_both_fail() -> None:
    """Test parallel execution with non-required stage where both fail."""
    class FailingParallelPrimaryBoth(IRAGStrategy):
        """Primary strategy that fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Primary failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class FailingParallelFallbackBoth(IRAGStrategy):
        """Fallback strategy that also fails."""

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve - raises error."""
            raise RuntimeError("Fallback failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    from rag_factory.pipeline import PipelineStage

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    stage = PipelineStage(
        strategy=FailingParallelPrimaryBoth(),
        name="both_fail",
        fallback=FailingParallelFallbackBoth(),
        required=False
    )
    pipeline.stages.append(stage)

    # Should not raise, just continue
    result = await pipeline.aexecute("query", top_k=5)
    assert len(result.errors) == 2  # Both primary and fallback errors
