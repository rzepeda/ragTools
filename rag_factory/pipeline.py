"""
Strategy Pipeline for orchestrating multiple RAG strategies.

This module provides the StrategyPipeline class that allows combining multiple
RAG strategies in various execution modes (sequential, parallel, cascade).
It supports error handling, fallback strategies, result merging, and
performance monitoring.

Example usage:
    >>> from rag_factory.pipeline import StrategyPipeline, ExecutionMode
    >>> from rag_factory.factory import RAGFactory
    >>>
    >>> # Create pipeline programmatically
    >>> pipeline = StrategyPipeline(mode=ExecutionMode.SEQUENTIAL)
    >>> pipeline.add_stage(strategy1, "stage1")
    >>> pipeline.add_stage(strategy2, "stage2", fallback=fallback_strategy)
    >>>
    >>> # Execute pipeline
    >>> result = pipeline.execute("What is RAG?", top_k=5)
    >>> print(f"Found {len(result.final_results)} results")
    >>>
    >>> # Or create from configuration
    >>> config = {
    ...     "mode": "parallel",
    ...     "stages": [
    ...         {"strategy": "dense", "name": "dense_retrieval"},
    ...         {"strategy": "sparse", "name": "sparse_retrieval"}
    ...     ]
    ... }
    >>> pipeline = StrategyPipeline.from_config(config)
    >>> result = await pipeline.aexecute("query", top_k=10)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from rag_factory.factory import RAGFactory
from rag_factory.strategies.base import Chunk, IRAGStrategy


class ExecutionMode(Enum):
    """Pipeline execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CASCADE = "cascade"


@dataclass
class PipelineStage:
    """
    Represents a single stage in the pipeline.

    Attributes:
        strategy: The RAG strategy to execute in this stage
        name: Unique identifier for this stage
        config: Configuration parameters for this stage
        fallback: Optional fallback strategy if primary fails
        required: Whether this stage must succeed for pipeline to continue
    """

    strategy: IRAGStrategy
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    fallback: Optional[IRAGStrategy] = None
    required: bool = True


@dataclass
class PipelineResult:
    """
    Results from pipeline execution.

    Attributes:
        final_results: Merged and deduplicated results from all stages
        stage_results: Results from each individual stage
        performance_metrics: Execution time metrics for each stage
        errors: List of errors that occurred during execution
        metadata: Additional metadata about the pipeline execution
    """

    final_results: List[Chunk]
    stage_results: Dict[str, List[Chunk]]
    performance_metrics: Dict[str, float]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class StrategyPipeline:
    """
    Orchestrates multiple RAG strategies in a pipeline.

    This class provides a flexible framework for combining multiple RAG
    strategies with different execution modes. It supports:
    - Sequential execution (strategies run one after another)
    - Parallel execution (strategies run concurrently)
    - Error handling with fallback strategies
    - Result merging and deduplication
    - Performance monitoring

    Attributes:
        stages: List of pipeline stages to execute
        mode: Execution mode (sequential, parallel, cascade)
        metrics: Performance metrics for each stage
        errors: List of errors encountered during execution

    Example:
        >>> pipeline = StrategyPipeline(mode=ExecutionMode.PARALLEL)
        >>> pipeline.add_stage(strategy1, "dense")
        >>> pipeline.add_stage(strategy2, "sparse")
        >>> result = await pipeline.aexecute("query", top_k=5)
        >>> print(f"Total execution time: {result.performance_metrics['total']}")
    """

    def __init__(
        self,
        stages: Optional[List[PipelineStage]] = None,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            stages: Optional list of pre-configured pipeline stages
            mode: Execution mode for the pipeline (default: SEQUENTIAL)
        """
        self.stages = stages or []
        self.mode = mode
        self.metrics: Dict[str, float] = {}
        self.errors: List[Dict[str, Any]] = []

    def add_stage(
        self,
        strategy: IRAGStrategy,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        fallback: Optional[IRAGStrategy] = None
    ) -> 'StrategyPipeline':
        """
        Add a strategy stage to the pipeline.

        Args:
            strategy: The RAG strategy to add
            name: Unique identifier for this stage
            config: Optional configuration parameters
            fallback: Optional fallback strategy if primary fails

        Returns:
            StrategyPipeline: Self for method chaining

        Example:
            >>> pipeline = (StrategyPipeline()
            ...     .add_stage(strategy1, "stage1")
            ...     .add_stage(strategy2, "stage2", fallback=fallback2))
        """
        stage = PipelineStage(
            strategy=strategy,
            name=name,
            config=config or {},
            fallback=fallback
        )
        self.stages.append(stage)
        return self

    def execute(self, query: str, top_k: int = 5) -> PipelineResult:
        """
        Execute the pipeline synchronously.

        Args:
            query: The search query
            top_k: Number of top results to return per strategy

        Returns:
            PipelineResult: Combined results from all stages

        Example:
            >>> result = pipeline.execute("What is RAG?", top_k=5)
            >>> print(f"Found {len(result.final_results)} results")
        """
        if self.mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(query, top_k)
        if self.mode == ExecutionMode.PARALLEL:
            return asyncio.run(self._execute_parallel(query, top_k))
        raise NotImplementedError(f"Mode {self.mode} not implemented")

    async def aexecute(self, query: str, top_k: int = 5) -> PipelineResult:
        """
        Execute the pipeline asynchronously.

        Args:
            query: The search query
            top_k: Number of top results to return per strategy

        Returns:
            PipelineResult: Combined results from all stages

        Example:
            >>> result = await pipeline.aexecute("What is RAG?", top_k=5)
            >>> print(f"Found {len(result.final_results)} results")
        """
        if self.mode == ExecutionMode.SEQUENTIAL:
            return await self._aexecute_sequential(query, top_k)
        if self.mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(query, top_k)
        raise NotImplementedError(f"Mode {self.mode} not implemented")

    def _execute_sequential(self, query: str, top_k: int) -> PipelineResult:
        """
        Execute strategies sequentially.

        Args:
            query: The search query
            top_k: Number of results to retrieve per strategy

        Returns:
            PipelineResult: Combined results from sequential execution
        """
        results: List[Chunk] = []
        stage_results: Dict[str, List[Chunk]] = {}
        start_time = time.time()

        for stage in self.stages:
            try:
                stage_start = time.time()
                stage_output = stage.strategy.retrieve(query, top_k)
                stage_duration = time.time() - stage_start

                self.metrics[stage.name] = stage_duration
                stage_results[stage.name] = stage_output
                results.extend(stage_output)

            except Exception as e:  # pylint: disable=broad-exception-caught
                self._handle_stage_error(stage, e)
                if stage.fallback:
                    # Execute fallback
                    try:
                        stage_output = stage.fallback.retrieve(query, top_k)
                        results.extend(stage_output)
                    except Exception as fallback_error:  # pylint: disable=broad-exception-caught
                        if stage.required:
                            raise
                        self._handle_stage_error(stage, fallback_error)
                elif stage.required:
                    raise

        total_duration = time.time() - start_time
        self.metrics["total"] = total_duration

        return PipelineResult(
            final_results=self._merge_results(results),
            stage_results=stage_results,
            performance_metrics=self.metrics.copy(),
            errors=self.errors.copy(),
            metadata={"mode": self.mode.value, "stages": len(self.stages)}
        )

    async def _aexecute_sequential(
        self,
        query: str,
        top_k: int
    ) -> PipelineResult:
        """
        Execute strategies sequentially (async version).

        Args:
            query: The search query
            top_k: Number of results to retrieve per strategy

        Returns:
            PipelineResult: Combined results from sequential execution
        """
        results: List[Chunk] = []
        stage_results: Dict[str, List[Chunk]] = {}
        start_time = time.time()

        for stage in self.stages:
            try:
                stage_start = time.time()
                stage_output = await stage.strategy.aretrieve(query, top_k)
                stage_duration = time.time() - stage_start

                self.metrics[stage.name] = stage_duration
                stage_results[stage.name] = stage_output
                results.extend(stage_output)

            except Exception as e:  # pylint: disable=broad-exception-caught
                self._handle_stage_error(stage, e)
                if stage.fallback:
                    # Execute fallback
                    try:
                        stage_output = await stage.fallback.aretrieve(
                            query,
                            top_k
                        )
                        results.extend(stage_output)
                    except Exception as fallback_error:  # pylint: disable=broad-exception-caught
                        if stage.required:
                            raise
                        self._handle_stage_error(stage, fallback_error)
                elif stage.required:
                    raise

        total_duration = time.time() - start_time
        self.metrics["total"] = total_duration

        return PipelineResult(
            final_results=self._merge_results(results),
            stage_results=stage_results,
            performance_metrics=self.metrics.copy(),
            errors=self.errors.copy(),
            metadata={"mode": self.mode.value, "stages": len(self.stages)}
        )

    async def _execute_parallel(
        self,
        query: str,
        top_k: int
    ) -> PipelineResult:
        """
        Execute strategies in parallel.

        Args:
            query: The search query
            top_k: Number of results to retrieve per strategy

        Returns:
            PipelineResult: Combined results from parallel execution
        """
        start_time = time.time()
        tasks = []

        for stage in self.stages:
            task = self._execute_stage_async(stage, query, top_k)
            tasks.append(task)

        stage_results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results: List[Chunk] = []
        stage_results: Dict[str, List[Chunk]] = {}

        for stage, result in zip(self.stages, stage_results_list):
            if isinstance(result, Exception):
                self._handle_stage_error(stage, result)
                if stage.fallback:
                    try:
                        fallback_result = await stage.fallback.aretrieve(
                            query,
                            top_k
                        )
                        results.extend(fallback_result)
                    except Exception as fallback_error:  # pylint: disable=broad-exception-caught
                        if stage.required:
                            # Re-raise the original error for required stages
                            raise result from fallback_error
                        self._handle_stage_error(stage, fallback_error)
                elif stage.required:
                    raise result
            else:
                # result is List[Chunk] here after isinstance check
                chunk_list = cast(List[Chunk], result)
                stage_results[stage.name] = chunk_list
                results.extend(chunk_list)

        total_duration = time.time() - start_time
        self.metrics["total"] = total_duration

        return PipelineResult(
            final_results=self._merge_results(results),
            stage_results=stage_results,
            performance_metrics=self.metrics.copy(),
            errors=self.errors.copy(),
            metadata={"mode": self.mode.value, "stages": len(self.stages)}
        )

    async def _execute_stage_async(
        self,
        stage: PipelineStage,
        query: str,
        top_k: int
    ) -> List[Chunk]:
        """
        Execute a single stage asynchronously.

        Args:
            stage: The pipeline stage to execute
            query: The search query
            top_k: Number of results to retrieve

        Returns:
            List[Chunk]: Results from the strategy
        """
        stage_start = time.time()
        result = await stage.strategy.aretrieve(query, top_k)
        stage_duration = time.time() - stage_start
        self.metrics[stage.name] = stage_duration
        return result

    def _merge_results(self, results: List[Chunk]) -> List[Chunk]:
        """
        Merge and deduplicate results from multiple strategies.

        Results are deduplicated by text content and sorted by score
        in descending order (highest score first).

        Args:
            results: List of chunks from all strategies

        Returns:
            List[Chunk]: Merged and deduplicated chunks sorted by score
        """
        # Deduplicate by text content
        seen_texts = set()
        merged = []

        for chunk in sorted(results, key=lambda x: x.score, reverse=True):
            if chunk.text not in seen_texts:
                seen_texts.add(chunk.text)
                merged.append(chunk)

        return merged

    def _handle_stage_error(
        self,
        stage: PipelineStage,
        error: Exception
    ) -> None:
        """
        Handle errors from strategy execution.

        Args:
            stage: The stage that encountered an error
            error: The exception that was raised
        """
        error_info = {
            "stage": stage.name,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": time.time()
        }
        self.errors.append(error_info)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StrategyPipeline':
        """
        Create pipeline from configuration dict.

        The configuration should have the following structure:
        {
            "mode": "sequential" | "parallel" | "cascade",
            "stages": [
                {
                    "strategy": "strategy_name",
                    "name": "stage_name",
                    "config": {...}
                },
                ...
            ]
        }

        Args:
            config: Configuration dictionary

        Returns:
            StrategyPipeline: Configured pipeline instance

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "mode": "parallel",
            ...     "stages": [
            ...         {"strategy": "dense", "name": "dense_stage"},
            ...         {"strategy": "sparse", "name": "sparse_stage"}
            ...     ]
            ... }
            >>> pipeline = StrategyPipeline.from_config(config)
        """
        mode = ExecutionMode(config.get("mode", "sequential"))
        pipeline = cls(mode=mode)

        factory = RAGFactory()

        for stage_config in config.get("stages", []):
            strategy_name = stage_config["strategy"]
            strategy = factory.create_strategy(
                strategy_name,
                stage_config.get("config")
            )

            pipeline.add_stage(
                strategy=strategy,
                name=stage_config.get("name", strategy_name),
                config=stage_config.get("config", {})
            )

        return pipeline
