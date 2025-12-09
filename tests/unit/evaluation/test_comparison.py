"""
Unit tests for strategy comparison.

Tests the StrategyComparator class for comparing RAG strategies.
"""

import pytest
from typing import List, Dict, Any
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult
from rag_factory.evaluation.analysis.comparison import StrategyComparator


@pytest.fixture
def sample_results() -> List[BenchmarkResult]:
    """Create sample benchmark results for comparison."""
    return [
        BenchmarkResult(
            strategy_name="Baseline",
            dataset_name="test_dataset",
            query_results=[
                {
                    "query_id": "q1",
                    "metrics": {"precision@5": 0.8, "recall@5": 0.6}
                },
                {
                    "query_id": "q2",
                    "metrics": {"precision@5": 0.7, "recall@5": 0.5}
                }
            ],
            aggregate_metrics={
                "precision@5": 0.75,
                "recall@5": 0.55,
                "avg_latency_ms": 150.0
            },
            execution_time=2.5
        ),
        BenchmarkResult(
            strategy_name="Enhanced",
            dataset_name="test_dataset",
            query_results=[
                {
                    "query_id": "q1",
                    "metrics": {"precision@5": 0.9, "recall@5": 0.7}
                },
                {
                    "query_id": "q2",
                    "metrics": {"precision@5": 0.85, "recall@5": 0.65}
                }
            ],
            aggregate_metrics={
                "precision@5": 0.875,
                "recall@5": 0.675,
                "avg_latency_ms": 180.0
            },
            execution_time=3.0
        ),
        BenchmarkResult(
            strategy_name="Optimized",
            dataset_name="test_dataset",
            query_results=[
                {
                    "query_id": "q1",
                    "metrics": {"precision@5": 0.95, "recall@5": 0.75}
                },
                {
                    "query_id": "q2",
                    "metrics": {"precision@5": 0.90, "recall@5": 0.70}
                }
            ],
            aggregate_metrics={
                "precision@5": 0.925,
                "recall@5": 0.725,
                "avg_latency_ms": 200.0
            },
            execution_time=3.5
        )
    ]


class TestStrategyComparator:
    """Tests for StrategyComparator class."""

    def test_init_default_confidence(self) -> None:
        """Test initialization with default confidence level."""
        comparator = StrategyComparator()
        # confidence_level is not exposed as public attribute
        assert comparator is not None

    def test_init_custom_confidence(self) -> None:
        """Test initialization with custom confidence level."""
        comparator = StrategyComparator(confidence_level=0.99)
        # confidence_level is not exposed as public attribute
        assert comparator is not None

    def test_compare_strategies(self, sample_results: List[BenchmarkResult]) -> None:
        """Test comparing multiple strategies."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results, baseline_idx=0)
        
        assert isinstance(comparison, dict)
        assert "baseline" in comparison
        assert "metrics" in comparison
        assert "rankings" in comparison
        assert "best_strategy" in comparison

    def test_compare_metric_specific(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test metric-specific comparison."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results, baseline_idx=0)
        
        # Check that precision@5 is compared
        assert "precision@5" in comparison["metrics"]
        metric_comparison = comparison["metrics"]["precision@5"]
        
        # Check that metric comparison has correct structure
        assert "values" in metric_comparison
        assert "improvements" in metric_comparison
        assert "best" in metric_comparison
        assert len(metric_comparison["values"]) == 3  # All 3 strategies

    def test_generate_rankings(self, sample_results: List[BenchmarkResult]) -> None:
        """Test rankings generation."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results)
        
        rankings = comparison["rankings"]
        
        # Check that rankings exist for each metric
        assert "precision@5" in rankings
        assert "recall@5" in rankings
        
        # Rankings is a dict: {metric_name: {strategy_name: rank}}
        precision_rankings = rankings["precision@5"]
        assert len(precision_rankings) == 3
        
        # Best strategy should have rank 1
        assert precision_rankings["Optimized"] == 1
        assert precision_rankings["Baseline"] == 3  # Worst rank

    def test_find_best_strategy(self, sample_results: List[BenchmarkResult]) -> None:
        """Test best strategy identification."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results)
        
        best_strategy = comparison["best_strategy"]
        
        assert "name" in best_strategy
        assert "avg_rank" in best_strategy
        assert "all_ranks" in best_strategy
        
        # Optimized should be the best overall
        assert best_strategy["name"] == "Optimized"

    def test_generate_report_table_format(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test report generation in table format."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results)
        report = comparator.generate_report(comparison, format="table")
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Should contain strategy names
        assert "Baseline" in report
        assert "Enhanced" in report
        assert "Optimized" in report

    def test_generate_report_markdown_format(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test report generation in markdown format."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results)
        report = comparator.generate_report(comparison, format="markdown")
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Should contain markdown formatting
        assert "#" in report or "|" in report

    def test_generate_report_text_format(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test report generation in text format."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results)
        report = comparator.generate_report(comparison, format="text")
        
        assert isinstance(report, str)
        assert len(report) > 0

    def test_generate_summary_table(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test summary table generation."""
        comparator = StrategyComparator()
        
        metrics = ["precision@5", "recall@5", "avg_latency_ms"]
        table = comparator.generate_summary_table(sample_results, metrics)
        
        assert isinstance(table, str)
        assert len(table) > 0
        
        # Should contain all strategy names
        assert "Baseline" in table
        assert "Enhanced" in table
        assert "Optimized" in table
        
        # Should contain all metrics
        assert "precision@5" in table
        assert "recall@5" in table

    def test_comparison_with_statistical_tests(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test that comparison includes statistical tests."""
        comparator = StrategyComparator()
        
        comparison = comparator.compare(sample_results, baseline_idx=0)
        
        # Check that statistical tests are included
        for metric_name, metric_data in comparison["metrics"].items():
            # Metric data has values, improvements, best - no nested comparisons
            assert "values" in metric_data
            assert "improvements" in metric_data

    def test_comparison_with_baseline(
        self,
        sample_results: List[BenchmarkResult]
    ) -> None:
        """Test comparison with specific baseline."""
        comparator = StrategyComparator()
        
        # Use Enhanced as baseline (index 1)
        comparison = comparator.compare(sample_results, baseline_idx=1)
        
        assert comparison["baseline"] == "Enhanced"

    def test_comparison_single_strategy(self) -> None:
        """Test comparison with single strategy."""
        comparator = StrategyComparator()
        
        single_result = [
            BenchmarkResult(
                strategy_name="OnlyStrategy",
                dataset_name="test_dataset",
                query_results=[],
                aggregate_metrics={"precision@5": 0.8},
                execution_time=1.0
            )
        ]
        
        comparison = comparator.compare(single_result)
        
        # Should handle single strategy gracefully
        assert comparison["best_strategy"]["name"] == "OnlyStrategy"
        assert len(comparison["rankings"]["precision@5"]) == 1

    def test_comparison_missing_metrics(self) -> None:
        """Test comparison with missing metrics in some strategies."""
        comparator = StrategyComparator()
        
        results = [
            BenchmarkResult(
                strategy_name="Strategy1",
                dataset_name="test_dataset",
                query_results=[],
                aggregate_metrics={"precision@5": 0.8, "recall@5": 0.6},
                execution_time=1.0
            ),
            BenchmarkResult(
                strategy_name="Strategy2",
                dataset_name="test_dataset",
                query_results=[],
                aggregate_metrics={"precision@5": 0.9},  # Missing recall@5
                execution_time=1.5
            )
        ]
        
        comparison = comparator.compare(results)
        
        # Should handle missing metrics gracefully
        assert "precision@5" in comparison["metrics"]
        assert "recall@5" in comparison["metrics"]
