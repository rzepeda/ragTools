"""
Integration tests for Strategy Pipeline.

This module contains integration tests that verify the complete pipeline
functionality with multiple strategies, performance comparisons, error recovery,
and configuration loading from files.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from rag_factory.factory import RAGFactory
from rag_factory.pipeline import ExecutionMode, StrategyPipeline
from rag_factory.strategies.base import Chunk, IRAGStrategy


# Helper strategy classes for integration tests
class TestStrategy(IRAGStrategy):
    """Base test strategy with configurable behavior."""

    def __init__(self, name: str, delay: float = 0.0) -> None:
        """Initialize test strategy."""
        self.strategy_name = name
        self.delay = delay
        self.config: Any = None

    def requires_services(self):
        """Declare required services."""
        from rag_factory.services.dependencies import ServiceDependency
        return set()

    def initialize(self, config: Any) -> None:
        """Initialize the strategy."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
        """Prepare data (not used in integration tests)."""
        return None

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve chunks with optional delay."""
        if self.delay > 0:
            time.sleep(self.delay)
        return [
            Chunk(
                f"{self.strategy_name}-{i}",
                {},
                0.9 - i * 0.1,
                f"doc{i}",
                f"c{i}"
            )
            for i in range(2)
        ]

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve chunks with optional delay."""
        import asyncio
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return [
            Chunk(
                f"{self.strategy_name}-{i}",
                {},
                0.9 - i * 0.1,
                f"doc{i}",
                f"c{i}"
            )
            for i in range(2)
        ]

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query."""
        return f"Answer from {self.strategy_name}"


# IS3.1: Multi-Strategy Pipeline Integration
@pytest.mark.integration
def test_three_strategy_pipeline() -> None:
    """Test pipeline with 3 different strategies."""
    strategy1 = TestStrategy("S1")
    strategy2 = TestStrategy("S2")
    strategy3 = TestStrategy("S3")

    pipeline = (StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
                .add_stage(strategy1, "strategy1")
                .add_stage(strategy2, "strategy2")
                .add_stage(strategy3, "strategy3"))

    result = pipeline.execute("test query", top_k=5)

    # Should have results from all 3 strategies
    assert len(result.stage_results) == 3
    # Total unique results (2 per strategy)
    assert len(result.final_results) == 6


# IS3.2: Sequential vs Parallel Performance
@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_faster_than_sequential() -> None:
    """Test parallel execution is faster for independent strategies."""
    # Create slow strategies with delays
    strategy1 = TestStrategy("S1", delay=0.1)
    strategy2 = TestStrategy("S2", delay=0.1)

    # Sequential pipeline
    seq_pipeline = (StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
                    .add_stage(strategy1, "s1")
                    .add_stage(strategy2, "s2"))

    start = time.time()
    seq_result = seq_pipeline.execute("query", top_k=5)
    seq_time = time.time() - start

    # Parallel pipeline (need new instances)
    strategy3 = TestStrategy("S1", delay=0.1)
    strategy4 = TestStrategy("S2", delay=0.1)

    par_pipeline = (StrategyPipeline(mode=ExecutionMode.PARALLEL)
                    .add_stage(strategy3, "s1")
                    .add_stage(strategy4, "s2"))

    start = time.time()
    par_result = await par_pipeline.aexecute("query", top_k=5)
    par_time = time.time() - start

    # Both should have the same results
    assert len(seq_result.final_results) == len(par_result.final_results)

    # Parallel should be significantly faster
    # Sequential should take ~0.2s (0.1 + 0.1)
    # Parallel should take ~0.1s (max of 0.1, 0.1)
    assert par_time < seq_time * 0.7  # At least 30% faster


# IS3.3: Error Recovery Integration
@pytest.mark.integration
def test_pipeline_continues_after_non_critical_failure() -> None:
    """Test pipeline continues when non-required strategy fails."""
    class OptionalStrategy(IRAGStrategy):
        """Strategy that always fails."""

        def requires_services(self):
            """Declare required services."""
            from rag_factory.services.dependencies import ServiceDependency
            return set()

        def initialize(self, config: Any) -> None:
            """Initialize."""
            pass

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve - raises error."""
            raise RuntimeError("Optional failed")

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    required_strategy = TestStrategy("Required")

    pipeline = StrategyPipeline()
    # Add optional failing stage
    from rag_factory.pipeline import PipelineStage
    optional_stage = PipelineStage(
        strategy=OptionalStrategy(),
        name="optional",
        required=False
    )
    pipeline.stages.append(optional_stage)
    # Add required working stage
    pipeline.add_stage(required_strategy, "required")

    result = pipeline.execute("query", top_k=5)

    # Pipeline should complete
    assert len(result.final_results) == 2  # From required strategy
    assert len(result.errors) == 1
    assert result.errors[0]["stage"] == "optional"


# IS3.4: Real Configuration File Integration
@pytest.mark.integration
def test_load_pipeline_from_yaml(tmp_path: Path) -> None:
    """Test loading pipeline from YAML configuration."""

    # Create factory-compatible strategies
    class StrategyA(IRAGStrategy):
        """Factory-compatible strategy A."""

        def requires_services(self):
            """Declare required services."""
            from rag_factory.services.dependencies import ServiceDependency
            return set()

        def initialize(self, config: Any) -> None:
            """Initialize."""
            self.config = config

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("A", {}, 0.9, "doc1", "c1")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return [Chunk("A", {}, 0.9, "doc1", "c1")]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class StrategyB(IRAGStrategy):
        """Factory-compatible strategy B."""

        def requires_services(self):
            """Declare required services."""
            from rag_factory.services.dependencies import ServiceDependency
            return set()

        def initialize(self, config: Any) -> None:
            """Initialize."""
            self.config = config

        def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
            """Prepare data."""
            return None

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return [Chunk("B", {}, 0.8, "doc2", "c2")]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return [Chunk("B", {}, 0.8, "doc2", "c2")]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    # Register test strategies
    RAGFactory.register_strategy("strategy_a", StrategyA, override=True)
    RAGFactory.register_strategy("strategy_b", StrategyB, override=True)

    # Create config file
    config_file = tmp_path / "pipeline_config.yaml"
    config_content = """
mode: sequential
stages:
  - strategy: strategy_a
    name: stage_a
    config:
      chunk_size: 512
  - strategy: strategy_b
    name: stage_b
    config:
      chunk_size: 1024
"""
    config_file.write_text(config_content)

    # Load config
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = StrategyPipeline.from_config(config)

    result = pipeline.execute("query", top_k=5)

    assert len(pipeline.stages) == 2
    assert pipeline.mode == ExecutionMode.SEQUENTIAL
    assert len(result.final_results) == 2
    assert len(result.stage_results) == 2


# Additional integration test for async sequential execution
@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_sequential_execution() -> None:
    """Test async sequential execution path."""
    strategy1 = TestStrategy("Async1", delay=0.05)
    strategy2 = TestStrategy("Async2", delay=0.05)

    pipeline = (StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
                .add_stage(strategy1, "stage1")
                .add_stage(strategy2, "stage2"))

    start_time = time.time()
    result = await pipeline.aexecute("query", top_k=5)
    total_time = time.time() - start_time

    # Should have results from both stages
    assert len(result.final_results) == 4
    assert len(result.stage_results) == 2

    # Sequential should take approximately sum of delays
    assert total_time >= 0.1  # At least 0.1 seconds

    # Check performance metrics
    assert "stage1" in result.performance_metrics
    assert "stage2" in result.performance_metrics
    assert "total" in result.performance_metrics


# Integration test for fallback with async execution
@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_fallback_execution() -> None:
    """Test fallback strategy in async mode."""
    class FailingAsyncStrategy(IRAGStrategy):
        """Strategy that always fails in async mode."""

        def requires_services(self):
            """Declare required services."""
            from rag_factory.services.dependencies import ServiceDependency
            return set()

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
            raise RuntimeError("Async primary failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    fallback_strategy = TestStrategy("Fallback")
    primary_strategy = FailingAsyncStrategy()

    pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    pipeline.add_stage(primary_strategy, "primary", fallback=fallback_strategy)

    result = await pipeline.aexecute("query", top_k=5)

    # Should have results from fallback
    assert len(result.final_results) == 2
    assert any("Fallback" in chunk.text for chunk in result.final_results)


# Integration test for parallel execution with mixed success/failure
@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_execution_with_failures() -> None:
    """Test parallel execution handles mixed success and failure."""
    class FailingParallelStrategy(IRAGStrategy):
        """Strategy that fails in parallel execution."""

        def requires_services(self):
            """Declare required services."""
            from rag_factory.services.dependencies import ServiceDependency
            return set()

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
            raise RuntimeError("Parallel strategy failed")

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    working_strategy = TestStrategy("Working")
    failing_strategy = FailingParallelStrategy()

    pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
    # Add non-required failing stage
    from rag_factory.pipeline import PipelineStage
    failing_stage = PipelineStage(
        strategy=failing_strategy,
        name="failing",
        required=False
    )
    pipeline.stages.append(failing_stage)
    # Add working stage
    pipeline.add_stage(working_strategy, "working")

    result = await pipeline.aexecute("query", top_k=5)

    # Should have results from working strategy only
    assert len(result.final_results) == 2
    assert len(result.errors) == 1
    assert result.errors[0]["stage"] == "failing"
