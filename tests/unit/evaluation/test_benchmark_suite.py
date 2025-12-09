"""
Unit tests for benchmark suite.

Tests the BenchmarkRunner class for running evaluations on RAG strategies.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch
from rag_factory.evaluation.benchmarks.runner import BenchmarkRunner, BenchmarkResult
from rag_factory.evaluation.benchmarks.config import BenchmarkConfig
from rag_factory.evaluation.datasets.schema import EvaluationDataset, EvaluationExample
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType


@pytest.fixture
def mock_metric() -> IMetric:
    """Create a mock metric."""
    metric = Mock(spec=IMetric)
    metric.name = "test_metric"
    metric.metric_type = MetricType.RETRIEVAL
    metric.compute.return_value = MetricResult(
        name="test_metric",
        value=0.85,
        metadata={}
    )
    return metric


@pytest.fixture
def mock_strategy() -> Mock:
    """Create a mock RAG strategy."""
    strategy = Mock()
    strategy.__class__.__name__ = "MockStrategy"
    
    # Mock retrieve to return simple documents
    def mock_retrieve(query: str, top_k: int = 5) -> List[Any]:
        docs = []
        for i in range(min(top_k, 3)):
            doc = Mock()
            doc.source_id = f"doc{i+1}"
            doc.content = f"Content for {query}"
            docs.append(doc)
        return docs
    
    strategy.retrieve = mock_retrieve
    return strategy


@pytest.fixture
def sample_dataset() -> EvaluationDataset:
    """Create a sample evaluation dataset."""
    examples = [
        EvaluationExample(
            query_id="q1",
            query="What is machine learning?",
            relevant_doc_ids=["doc1", "doc2"],
            ground_truth_answer="ML is...",
            relevance_scores={"doc1": 3, "doc2": 2}
        ),
        EvaluationExample(
            query_id="q2",
            query="What is AI?",
            relevant_doc_ids=["doc2", "doc3"],
            ground_truth_answer="AI is...",
            relevance_scores={"doc2": 3, "doc3": 2}
        )
    ]
    
    return EvaluationDataset(
        name="test_dataset",
        examples=examples,
        metadata={"version": "1.0"}
    )


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init_with_config(self, mock_metric: IMetric) -> None:
        """Test initialization with BenchmarkConfig."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        assert runner.config == config
        assert runner.metrics == [mock_metric]

    def test_init_with_metrics(self, mock_metric: IMetric) -> None:
        """Test initialization with metrics list."""
        runner = BenchmarkRunner(metrics=[mock_metric])
        
        assert runner.metrics == [mock_metric]
        assert isinstance(runner.config, BenchmarkConfig)

    def test_init_without_config_or_metrics(self) -> None:
        """Test initialization fails without config or metrics."""
        with pytest.raises(ValueError, match="Either config or metrics must be provided"):
            BenchmarkRunner()

    def test_run_benchmark_with_dataset(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test running benchmark with sample dataset."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(mock_strategy, sample_dataset, strategy_name="TestStrategy")
        
        assert isinstance(result, BenchmarkResult)
        assert result.strategy_name == "TestStrategy"
        assert result.dataset_name == "test_dataset"
        assert len(result.query_results) == 2
        assert "test_metric" in result.aggregate_metrics
        assert result.execution_time > 0

    def test_run_benchmark_collects_metrics(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test that benchmark collects metrics for each query."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(mock_strategy, sample_dataset)
        
        # Check that each query has metrics
        for query_result in result.query_results:
            assert "metrics" in query_result
            assert "test_metric" in query_result["metrics"]
            assert query_result["metrics"]["test_metric"] == 0.85

    def test_run_benchmark_aggregates_results(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test that benchmark aggregates results correctly."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(mock_strategy, sample_dataset)
        
        # Check aggregated metrics
        assert "test_metric" in result.aggregate_metrics
        assert result.aggregate_metrics["test_metric"] == 0.85
        assert "avg_latency_ms" in result.aggregate_metrics
        assert "min_latency_ms" in result.aggregate_metrics
        assert "max_latency_ms" in result.aggregate_metrics

    def test_compare_strategies(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test comparing multiple strategies."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        # Create two strategies
        strategy1 = mock_strategy
        strategy2 = Mock()
        strategy2.__class__.__name__ = "MockStrategy2"
        strategy2.retrieve = mock_strategy.retrieve
        
        results = runner.compare_strategies(
            strategies=[strategy1, strategy2],
            dataset=sample_dataset,
            strategy_names=["Strategy1", "Strategy2"]
        )
        
        assert len(results) == 2
        assert results[0].strategy_name == "Strategy1"
        assert results[1].strategy_name == "Strategy2"
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_compare_strategies_name_mismatch(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test that compare_strategies raises error on name mismatch."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        with pytest.raises(ValueError, match="Number of strategy names must match"):
            runner.compare_strategies(
                strategies=[mock_strategy, mock_strategy],
                dataset=sample_dataset,
                strategy_names=["Strategy1"]  # Only one name for two strategies
            )

    def test_benchmark_error_handling(
        self,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric
    ) -> None:
        """Test error handling for failed strategy execution."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        # Create a strategy that raises an error
        failing_strategy = Mock()
        failing_strategy.__class__.__name__ = "FailingStrategy"
        failing_strategy.retrieve.side_effect = Exception("Retrieval failed")
        
        result = runner.run(failing_strategy, sample_dataset)
        
        # Check that errors are recorded
        assert len(result.query_results) == 2
        for query_result in result.query_results:
            assert "error" in query_result
            assert query_result["error"] == "Retrieval failed"

    def test_save_checkpoint(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test checkpoint saving functionality."""
        config = BenchmarkConfig(
            metrics=[mock_metric],
            verbose=False,
            enable_checkpointing=True,
            checkpoint_interval=1,
            cache_dir=str(tmp_path)
        )
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(mock_strategy, sample_dataset, strategy_name="TestStrategy")
        
        # Checkpoint should be created and then deleted after successful completion
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_file = checkpoint_dir / "TestStrategy_test_dataset_checkpoint.json"
        
        # Checkpoint should be deleted after successful run
        assert not checkpoint_file.exists()
        assert result is not None

    def test_load_checkpoint(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test checkpoint loading functionality."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        # Create a checkpoint file
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_data = {
            "query_results": [
                {
                    "query_id": "q1",
                    "query": "Test query",
                    "latency_ms": 100.0,
                    "results_count": 3,
                    "retrieved_ids": ["doc1", "doc2", "doc3"],
                    "metrics": {"test_metric": 0.85}
                }
            ],
            "timestamp": 1234567890.0
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Load checkpoint
        loaded_data = runner._load_checkpoint(str(checkpoint_path))
        
        assert loaded_data["query_results"] == checkpoint_data["query_results"]
        assert loaded_data["timestamp"] == checkpoint_data["timestamp"]

    def test_resume_from_checkpoint(
        self,
        mock_strategy: Mock,
        sample_dataset: EvaluationDataset,
        mock_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test resuming benchmark from checkpoint."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        # Create a checkpoint with one query already processed
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_data = {
            "query_results": [
                {
                    "query_id": "q1",
                    "query": "What is machine learning?",
                    "latency_ms": 100.0,
                    "results_count": 3,
                    "retrieved_ids": ["doc1", "doc2", "doc3"],
                    "metrics": {"test_metric": 0.85}
                }
            ],
            "timestamp": 1234567890.0
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Resume from checkpoint
        result = runner.run(
            mock_strategy,
            sample_dataset,
            strategy_name="TestStrategy",
            resume_from=str(checkpoint_path)
        )
        
        # Should have results for both queries
        assert len(result.query_results) == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_benchmark_result_to_dict(self) -> None:
        """Test converting BenchmarkResult to dictionary."""
        result = BenchmarkResult(
            strategy_name="TestStrategy",
            dataset_name="test_dataset",
            query_results=[{"query_id": "q1", "metrics": {"precision": 0.8}}],
            aggregate_metrics={"precision": 0.8},
            execution_time=1.5,
            metadata={"test": "value"},
            config={"verbose": False}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["strategy_name"] == "TestStrategy"
        assert result_dict["dataset_name"] == "test_dataset"
        assert result_dict["execution_time"] == 1.5
        assert result_dict["metadata"]["test"] == "value"

    def test_benchmark_result_save_load(self, tmp_path: Path) -> None:
        """Test saving and loading BenchmarkResult."""
        result = BenchmarkResult(
            strategy_name="TestStrategy",
            dataset_name="test_dataset",
            query_results=[{"query_id": "q1", "metrics": {"precision": 0.8}}],
            aggregate_metrics={"precision": 0.8},
            execution_time=1.5,
            metadata={"test": "value"}
        )
        
        # Save result
        save_path = tmp_path / "result.json"
        result.save(str(save_path))
        
        assert save_path.exists()
        
        # Load result
        loaded_result = BenchmarkResult.load(str(save_path))
        
        assert loaded_result.strategy_name == result.strategy_name
        assert loaded_result.dataset_name == result.dataset_name
        assert loaded_result.execution_time == result.execution_time
        assert loaded_result.aggregate_metrics == result.aggregate_metrics

    def test_empty_dataset_handling(
        self,
        mock_strategy: Mock,
        mock_metric: IMetric
    ) -> None:
        """Test handling of empty dataset."""
        config = BenchmarkConfig(metrics=[mock_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        empty_dataset = EvaluationDataset(
            name="empty_dataset",
            examples=[],
            metadata={}
        )
        
        result = runner.run(mock_strategy, empty_dataset)
        
        assert len(result.query_results) == 0
        assert result.aggregate_metrics == {}
