"""
Integration tests for evaluation framework.

Tests end-to-end evaluation workflows including benchmarking, comparison,
statistical analysis, and export.
"""

import pytest
from pathlib import Path
from typing import List, Any
from unittest.mock import Mock
from rag_factory.evaluation.benchmarks.runner import BenchmarkRunner, BenchmarkResult
from rag_factory.evaluation.benchmarks.config import BenchmarkConfig
from rag_factory.evaluation.datasets.schema import EvaluationDataset, EvaluationExample
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType
from rag_factory.evaluation.analysis.comparison import StrategyComparator
from rag_factory.evaluation.analysis.statistics import StatisticalAnalyzer
from rag_factory.evaluation.exporters.csv_exporter import CSVExporter
from rag_factory.evaluation.exporters.json_exporter import JSONExporter


@pytest.fixture
def simple_metric() -> IMetric:
    """Create a simple mock metric for testing."""
    metric = Mock(spec=IMetric)
    metric.name = "precision@5"
    metric.metric_type = MetricType.RETRIEVAL
    
    def compute_metric(**kwargs: Any) -> MetricResult:
        # Simple precision calculation
        retrieved_ids = kwargs.get("retrieved_ids", [])
        relevant_ids = kwargs.get("relevant_ids", [])
        
        if not retrieved_ids:
            precision = 0.0
        else:
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / len(retrieved_ids)
        
        return MetricResult(
            name="precision@5",
            value=precision,
            metadata={}
        )
    
    metric.compute = compute_metric
    return metric


@pytest.fixture
def simple_strategy() -> Mock:
    """Create a simple mock strategy."""
    strategy = Mock()
    strategy.__class__.__name__ = "SimpleStrategy"
    
    def retrieve(query: str, top_k: int = 5) -> List[Any]:
        # Return different docs based on query
        docs = []
        if "machine learning" in query.lower():
            doc_ids = ["doc1", "doc2", "doc3"]
        elif "ai" in query.lower():
            doc_ids = ["doc2", "doc4", "doc5"]
        else:
            doc_ids = ["doc1", "doc3", "doc5"]
        
        for doc_id in doc_ids[:top_k]:
            doc = Mock()
            doc.source_id = doc_id
            doc.content = f"Content for {query}"
            docs.append(doc)
        
        return docs
    
    strategy.retrieve = retrieve
    return strategy


@pytest.fixture
def enhanced_strategy() -> Mock:
    """Create an enhanced mock strategy with better retrieval."""
    strategy = Mock()
    strategy.__class__.__name__ = "EnhancedStrategy"
    
    def retrieve(query: str, top_k: int = 5) -> List[Any]:
        # Return more relevant docs
        docs = []
        if "machine learning" in query.lower():
            doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        elif "ai" in query.lower():
            doc_ids = ["doc2", "doc3", "doc4", "doc5"]
        else:
            doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        
        for doc_id in doc_ids[:top_k]:
            doc = Mock()
            doc.source_id = doc_id
            doc.content = f"Enhanced content for {query}"
            docs.append(doc)
        
        return docs
    
    strategy.retrieve = retrieve
    return strategy


@pytest.fixture
def small_dataset() -> EvaluationDataset:
    """Create a small evaluation dataset."""
    examples = [
        EvaluationExample(
            query_id="q1",
            query="What is machine learning?",
            relevant_doc_ids=["doc1", "doc2", "doc3"],
            ground_truth_answer="ML is a subset of AI..."
        ),
        EvaluationExample(
            query_id="q2",
            query="What is AI?",
            relevant_doc_ids=["doc2", "doc3", "doc4"],
            ground_truth_answer="AI is artificial intelligence..."
        ),
        EvaluationExample(
            query_id="q3",
            query="What is deep learning?",
            relevant_doc_ids=["doc1", "doc3", "doc5"],
            ground_truth_answer="Deep learning uses neural networks..."
        )
    ]
    
    return EvaluationDataset(
        name="small_test_dataset",
        examples=examples,
        metadata={"version": "1.0", "size": "small"}
    )


class TestEvaluationIntegration:
    """Integration tests for evaluation framework."""

    def test_full_benchmark_workflow(
        self,
        simple_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test full benchmark workflow: run and export."""
        # Run benchmark
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(
            strategy=simple_strategy,
            dataset=small_dataset,
            strategy_name="TestStrategy"
        )
        
        assert result is not None
        assert len(result.query_results) == 3
        assert "precision@5" in result.aggregate_metrics
        
        # Export to JSON
        json_exporter = JSONExporter()
        json_path = tmp_path / "result.json"
        json_exporter.export(result, str(json_path))
        
        assert json_path.exists()
        
        # Export to CSV
        csv_exporter = CSVExporter()
        csv_path = tmp_path / "result.csv"
        csv_exporter.export(result, str(csv_path), include_query_details=False)
        
        assert csv_path.exists()

    def test_compare_multiple_strategies(
        self,
        simple_strategy: Mock,
        enhanced_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test comparing multiple strategies end-to-end."""
        # Run benchmarks for both strategies
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        results = runner.compare_strategies(
            strategies=[simple_strategy, enhanced_strategy],
            dataset=small_dataset,
            strategy_names=["Simple", "Enhanced"]
        )
        
        assert len(results) == 2
        assert results[0].strategy_name == "Simple"
        assert results[1].strategy_name == "Enhanced"
        
        # Compare strategies
        comparator = StrategyComparator()
        comparison = comparator.compare(results, baseline_idx=0)
        
        assert "rankings" in comparison
        assert "best_strategy" in comparison
        
        # Export comparison
        csv_exporter = CSVExporter()
        comparison_path = tmp_path / "comparison.csv"
        csv_exporter.export_comparison(results, str(comparison_path))
        
        assert comparison_path.exists()

    def test_statistical_analysis_workflow(
        self,
        simple_strategy: Mock,
        enhanced_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric
    ) -> None:
        """Test statistical analysis on benchmark results."""
        # Run benchmarks
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result1 = runner.run(simple_strategy, small_dataset, "Simple")
        result2 = runner.run(enhanced_strategy, small_dataset, "Enhanced")
        
        # Extract metric values for statistical analysis
        simple_values = [
            qr["metrics"]["precision@5"]
            for qr in result1.query_results
            if "metrics" in qr and "precision@5" in qr["metrics"]
        ]
        
        enhanced_values = [
            qr["metrics"]["precision@5"]
            for qr in result2.query_results
            if "metrics" in qr and "precision@5" in qr["metrics"]
        ]
        
        # Perform statistical analysis
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # T-test
        t_test_result = analyzer.paired_t_test(
            baseline_scores=simple_values,
            comparison_scores=enhanced_values,
            metric_name="precision@5"
        )
        
        assert t_test_result is not None
        assert t_test_result.p_value is not None
        
        # Confidence intervals
        ci_simple = analyzer.confidence_interval(simple_values)
        ci_enhanced = analyzer.confidence_interval(enhanced_values)
        
        assert len(ci_simple) == 2
        assert len(ci_enhanced) == 2

    def test_export_workflow_csv(
        self,
        simple_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test full workflow with CSV export."""
        # Run benchmark
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(simple_strategy, small_dataset, "TestStrategy")
        
        # Export summary
        exporter = CSVExporter()
        summary_path = tmp_path / "summary.csv"
        exporter.export(result, str(summary_path), include_query_details=False)
        
        assert summary_path.exists()
        
        # Export detailed
        detailed_path = tmp_path / "detailed.csv"
        exporter.export(result, str(detailed_path), include_query_details=True)
        
        assert detailed_path.exists()

    def test_export_workflow_json(
        self,
        simple_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test full workflow with JSON export."""
        # Run benchmark
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        result = runner.run(simple_strategy, small_dataset, "TestStrategy")
        
        # Export with details
        exporter = JSONExporter()
        detailed_path = tmp_path / "detailed.json"
        exporter.export(result, str(detailed_path), include_query_details=True)
        
        assert detailed_path.exists()
        
        # Export summary
        summary_path = tmp_path / "summary.json"
        exporter.export(result, str(summary_path), include_query_details=False)
        
        assert summary_path.exists()

    def test_comparison_report_generation(
        self,
        simple_strategy: Mock,
        enhanced_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric
    ) -> None:
        """Test generating comparison reports."""
        # Run benchmarks
        config = BenchmarkConfig(metrics=[simple_metric], verbose=False)
        runner = BenchmarkRunner(config=config)
        
        results = runner.compare_strategies(
            strategies=[simple_strategy, enhanced_strategy],
            dataset=small_dataset,
            strategy_names=["Baseline", "Enhanced"]
        )
        
        # Generate comparison
        comparator = StrategyComparator()
        comparison = comparator.compare(results, baseline_idx=0)
        
        # Generate reports in different formats
        table_report = comparator.generate_report(comparison, format="table")
        markdown_report = comparator.generate_report(comparison, format="markdown")
        text_report = comparator.generate_report(comparison, format="text")
        
        assert len(table_report) > 0
        assert len(markdown_report) > 0
        assert len(text_report) > 0
        
        # Generate summary table
        summary = comparator.generate_summary_table(
            results,
            metrics=["precision@5"]
        )
        
        assert len(summary) > 0
        assert "Baseline" in summary
        assert "Enhanced" in summary

    def test_checkpoint_resume_workflow(
        self,
        simple_strategy: Mock,
        small_dataset: EvaluationDataset,
        simple_metric: IMetric,
        tmp_path: Path
    ) -> None:
        """Test checkpoint and resume workflow."""
        # Configure with checkpointing
        config = BenchmarkConfig(
            metrics=[simple_metric],
            verbose=False,
            enable_checkpointing=True,
            checkpoint_interval=1,
            cache_dir=str(tmp_path)
        )
        
        runner = BenchmarkRunner(config=config)
        
        # Run benchmark (should create and clean up checkpoint)
        result = runner.run(
            strategy=simple_strategy,
            dataset=small_dataset,
            strategy_name="TestStrategy"
        )
        
        assert result is not None
        assert len(result.query_results) == 3
        
        # Checkpoint should be cleaned up after successful run
        checkpoint_dir = tmp_path / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.json"))
            assert len(checkpoint_files) == 0
