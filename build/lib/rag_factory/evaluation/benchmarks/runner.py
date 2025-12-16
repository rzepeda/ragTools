"""
Benchmark runner for evaluating RAG strategies.

This module provides the core benchmarking functionality for running
evaluations on RAG strategies and comparing results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
from pathlib import Path
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from rag_factory.evaluation.datasets.schema import EvaluationDataset, EvaluationExample
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType
from rag_factory.evaluation.benchmarks.config import BenchmarkConfig
from rag_factory.strategies.base import IRAGStrategy


@dataclass
class BenchmarkResult:
    """
    Result from a benchmark run.

    Attributes:
        strategy_name: Name of the evaluated strategy
        dataset_name: Name of the evaluation dataset
        query_results: List of per-query results
        aggregate_metrics: Aggregated metrics across all queries
        execution_time: Total execution time in seconds
        metadata: Additional metadata about the run
        config: Benchmark configuration used

    Example:
        >>> print(f"Strategy: {result.strategy_name}")
        >>> print(f"Precision@5: {result.aggregate_metrics['precision@5']:.3f}")
        >>> print(f"Execution time: {result.execution_time:.2f}s")
    """
    strategy_name: str
    dataset_name: str
    query_results: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "dataset_name": self.dataset_name,
            "query_results": self.query_results,
            "aggregate_metrics": self.aggregate_metrics,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "config": self.config
        }

    def save(self, path: str) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkResult':
        """
        Load results from JSON file.

        Args:
            path: Input file path

        Returns:
            BenchmarkResult instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkRunner:
    """
    Run benchmarks on RAG strategies.

    Features:
    - Evaluate multiple strategies on datasets
    - Track progress with progress bars
    - Compute multiple metrics
    - Cache and checkpoint results
    - Aggregate metrics across queries

    Example:
        >>> from rag_factory.evaluation.metrics import PrecisionAtK, RecallAtK
        >>> config = BenchmarkConfig(
        ...     metrics=[PrecisionAtK(k=5), RecallAtK(k=5)],
        ...     verbose=True
        ... )
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.run(strategy, dataset, strategy_name="MyStrategy")
        >>> print(results.aggregate_metrics)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None, metrics: Optional[List[IMetric]] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration (if provided, metrics param is ignored)
            metrics: List of metrics to compute (used if config is None)

        Raises:
            ValueError: If neither config nor metrics are provided
        """
        if config is not None:
            self.config = config
            self.metrics = config.metrics
        elif metrics is not None:
            self.metrics = metrics
            self.config = BenchmarkConfig(metrics=metrics)
        else:
            raise ValueError("Either config or metrics must be provided")

    def run(
        self,
        strategy: IRAGStrategy,
        dataset: EvaluationDataset,
        strategy_name: Optional[str] = None,
        resume_from: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Run benchmark on a strategy.

        Args:
            strategy: RAG strategy to evaluate
            dataset: Evaluation dataset
            strategy_name: Optional name for strategy (defaults to class name)
            resume_from: Optional path to checkpoint to resume from

        Returns:
            BenchmarkResult with all metrics

        Example:
            >>> results = runner.run(strategy, dataset, "MyStrategy")
            >>> print(f"Precision@5: {results.aggregate_metrics['precision@5']:.3f}")
        """
        strategy_name = strategy_name or strategy.__class__.__name__

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {strategy_name}")
            print(f"Dataset: {dataset.name} ({len(dataset)} examples)")
            print(f"Metrics: {[m.name for m in self.metrics]}")
            print(f"{'='*60}\n")

        start_time = time.time()
        query_results = []

        # Setup checkpoint if enabled
        checkpoint_path = None
        if self.config.enable_checkpointing:
            checkpoint_dir = Path(self.config.cache_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{strategy_name}_{dataset.name}_checkpoint.json"

        # Resume from checkpoint if available
        start_idx = 0
        if resume_from and Path(resume_from).exists():
            checkpoint_data = self._load_checkpoint(resume_from)
            query_results = checkpoint_data.get("query_results", [])
            start_idx = len(query_results)
            if self.config.verbose:
                print(f"Resuming from checkpoint: {start_idx} queries already processed")

        # Create progress bar if tqdm is available
        examples = dataset.examples[start_idx:]
        if TQDM_AVAILABLE and self.config.verbose:
            iterator = tqdm(examples, desc=f"Evaluating {strategy_name}", initial=start_idx, total=len(dataset))
        else:
            iterator = examples

        # Evaluate each query
        for idx, example in enumerate(iterator, start=start_idx):
            try:
                query_result = self._evaluate_query(strategy, example)
                query_results.append(query_result)

                # Checkpoint if enabled
                if (self.config.enable_checkpointing and
                    checkpoint_path and
                    (idx + 1) % self.config.checkpoint_interval == 0):
                    self._save_checkpoint(checkpoint_path, query_results)

            except Exception as e:
                if self.config.verbose:
                    print(f"\nError evaluating query {example.query_id}: {e}")
                # Store error in result
                query_results.append({
                    "query_id": example.query_id,
                    "query": example.query,
                    "error": str(e),
                    "metrics": {}
                })

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(query_results)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Clean up checkpoint if completed successfully
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        result = BenchmarkResult(
            strategy_name=strategy_name,
            dataset_name=dataset.name,
            query_results=query_results,
            aggregate_metrics=aggregate_metrics,
            execution_time=execution_time,
            metadata={
                "total_queries": len(dataset),
                "successful_queries": len([r for r in query_results if "error" not in r]),
                "failed_queries": len([r for r in query_results if "error" in r])
            },
            config=self.config.to_dict()
        )

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Benchmark completed in {execution_time:.2f}s")
            print(f"Aggregate Metrics:")
            for metric_name, value in aggregate_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            print(f"{'='*60}\n")

        return result

    def _evaluate_query(
        self,
        strategy: IRAGStrategy,
        example: EvaluationExample
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            strategy: RAG strategy
            example: Evaluation example

        Returns:
            Dictionary with query results and metrics
        """
        # Execute retrieval
        start_time = time.time()
        retrieved_docs = strategy.retrieve(example.query, top_k=self.config.top_k)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Extract document IDs from retrieved results
        retrieved_ids = [doc.source_id for doc in retrieved_docs]

        # Compute all metrics
        metric_results = {}
        for metric in self.metrics:
            try:
                # Different metrics require different inputs
                if metric.metric_type == MetricType.RETRIEVAL:
                    if metric.name.startswith('ndcg'):
                        # NDCG requires relevance scores
                        if example.relevance_scores:
                            result = metric.compute(
                                retrieved_ids=retrieved_ids,
                                relevance_scores=example.relevance_scores,
                                query_id=example.query_id
                            )
                        else:
                            # Skip if no relevance scores available
                            continue
                    else:
                        # Standard retrieval metrics
                        result = metric.compute(
                            retrieved_ids=retrieved_ids,
                            relevant_ids=example.relevant_doc_ids,
                            query_id=example.query_id
                        )
                    metric_results[result.name] = result.value

                elif metric.metric_type == MetricType.PERFORMANCE:
                    # Performance metrics use measured latency
                    result = metric.compute(
                        latency_ms=latency,
                        query_id=example.query_id
                    )
                    metric_results[result.name] = result.value

            except Exception as e:
                # Log metric computation error but continue
                if self.config.verbose:
                    print(f"Warning: Error computing {metric.name}: {e}")
                metric_results[metric.name] = None

        return {
            "query_id": example.query_id,
            "query": example.query,
            "latency_ms": latency,
            "results_count": len(retrieved_docs),
            "retrieved_ids": retrieved_ids[:10],  # Store top 10 for inspection
            "metrics": metric_results
        }

    def _aggregate_metrics(
        self,
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across all queries.

        Args:
            query_results: List of per-query results

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}

        # Filter out failed queries
        valid_results = [r for r in query_results if "error" not in r]
        if not valid_results:
            return aggregated

        # Get all metric names
        metric_names = set()
        for result in valid_results:
            metric_names.update(result["metrics"].keys())

        # Compute averages for each metric
        for metric_name in metric_names:
            values = [
                result["metrics"][metric_name]
                for result in valid_results
                if result["metrics"].get(metric_name) is not None
            ]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        # Add performance metrics
        latencies = [r["latency_ms"] for r in valid_results if "latency_ms" in r]
        if latencies:
            aggregated["avg_latency_ms"] = sum(latencies) / len(latencies)
            aggregated["min_latency_ms"] = min(latencies)
            aggregated["max_latency_ms"] = max(latencies)

            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            p50_idx = int(len(sorted_latencies) * 0.50)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)

            aggregated["p50_latency_ms"] = sorted_latencies[p50_idx]
            aggregated["p95_latency_ms"] = sorted_latencies[p95_idx]
            aggregated["p99_latency_ms"] = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]

        return aggregated

    def _save_checkpoint(self, path: Path, query_results: List[Dict[str, Any]]) -> None:
        """Save checkpoint to file."""
        checkpoint_data = {
            "query_results": query_results,
            "timestamp": time.time()
        }
        with open(path, 'w') as f:
            json.dump(checkpoint_data, f)

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from file."""
        with open(path, 'r') as f:
            return json.load(f)

    def compare_strategies(
        self,
        strategies: List[IRAGStrategy],
        dataset: EvaluationDataset,
        strategy_names: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Compare multiple strategies on the same dataset.

        Args:
            strategies: List of strategies to evaluate
            dataset: Evaluation dataset
            strategy_names: Optional list of strategy names

        Returns:
            List of BenchmarkResults, one per strategy

        Example:
            >>> results = runner.compare_strategies(
            ...     [strategy1, strategy2, strategy3],
            ...     dataset,
            ...     ["Baseline", "Enhanced", "Optimized"]
            ... )
            >>> for result in results:
            ...     print(f"{result.strategy_name}: {result.aggregate_metrics}")
        """
        if strategy_names and len(strategy_names) != len(strategies):
            raise ValueError("Number of strategy names must match number of strategies")

        results = []
        for idx, strategy in enumerate(strategies):
            name = strategy_names[idx] if strategy_names else f"Strategy_{idx+1}"
            result = self.run(strategy, dataset, strategy_name=name)
            results.append(result)

        return results
