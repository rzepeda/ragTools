"""
Strategy comparison tools.

This module provides utilities for comparing multiple RAG strategies
and generating comparison reports.
"""

from typing import List, Dict, Any
import numpy as np
from tabulate import tabulate
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult
from rag_factory.evaluation.analysis.statistics import StatisticalAnalyzer


class StrategyComparator:
    """
    Compare multiple RAG strategies.

    Provides tools for comparing benchmark results across strategies,
    identifying the best strategy, and generating comparison reports.

    Example:
        >>> comparator = StrategyComparator()
        >>> comparison = comparator.compare([result1, result2, result3])
        >>> print(comparator.generate_report(comparison))
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize strategy comparator.

        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.analyzer = StatisticalAnalyzer(confidence_level)

    def compare(
        self,
        results: List[BenchmarkResult],
        baseline_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.

        Args:
            results: List of benchmark results to compare
            baseline_idx: Index of baseline strategy (default: 0)

        Returns:
            Dictionary with comparison data

        Example:
            >>> comparison = comparator.compare([baseline_result, new_result])
            >>> print(comparison["summary"])
        """
        if not results:
            raise ValueError("At least one result required")
        if baseline_idx >= len(results):
            raise ValueError(f"baseline_idx {baseline_idx} out of range")

        baseline = results[baseline_idx]
        comparison_data = {
            "baseline": baseline.strategy_name,
            "strategies": [r.strategy_name for r in results],
            "metrics": {},
            "rankings": {},
            "best_strategy": {}
        }

        # Get all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.aggregate_metrics.keys())

        # Compare each metric
        for metric_name in sorted(all_metrics):
            metric_data = self._compare_metric(results, metric_name, baseline_idx)
            comparison_data["metrics"][metric_name] = metric_data

        # Generate rankings for each metric
        comparison_data["rankings"] = self._generate_rankings(results)

        # Find best overall strategy
        comparison_data["best_strategy"] = self._find_best_strategy(results)

        return comparison_data

    def _compare_metric(
        self,
        results: List[BenchmarkResult],
        metric_name: str,
        baseline_idx: int
    ) -> Dict[str, Any]:
        """Compare a specific metric across strategies."""
        metric_comparison = {
            "name": metric_name,
            "values": {},
            "improvements": {},
            "best": None
        }

        # Collect metric values
        baseline_value = results[baseline_idx].aggregate_metrics.get(metric_name)
        for result in results:
            value = result.aggregate_metrics.get(metric_name)
            metric_comparison["values"][result.strategy_name] = value

            # Calculate improvement over baseline
            if value is not None and baseline_value is not None and baseline_value != 0:
                improvement = ((value - baseline_value) / abs(baseline_value)) * 100
                metric_comparison["improvements"][result.strategy_name] = improvement

        # Find best strategy for this metric
        # Assuming higher is better for most metrics (except latency)
        is_lower_better = "latency" in metric_name.lower() or "cost" in metric_name.lower()
        valid_values = {
            name: val for name, val in metric_comparison["values"].items()
            if val is not None
        }

        if valid_values:
            if is_lower_better:
                best_name = min(valid_values, key=valid_values.get)
            else:
                best_name = max(valid_values, key=valid_values.get)
            metric_comparison["best"] = best_name

        return metric_comparison

    def _generate_rankings(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Dict[str, int]]:
        """Generate rankings for each metric."""
        rankings = {}

        # Get all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.aggregate_metrics.keys())

        # Rank each metric
        for metric_name in all_metrics:
            # Collect values
            strategy_values = []
            for result in results:
                value = result.aggregate_metrics.get(metric_name)
                if value is not None:
                    strategy_values.append((result.strategy_name, value))

            if not strategy_values:
                continue

            # Sort by value (descending for most metrics, ascending for latency/cost)
            is_lower_better = "latency" in metric_name.lower() or "cost" in metric_name.lower()
            strategy_values.sort(key=lambda x: x[1], reverse=not is_lower_better)

            # Assign ranks
            metric_rankings = {}
            for rank, (strategy_name, _) in enumerate(strategy_values, start=1):
                metric_rankings[strategy_name] = rank

            rankings[metric_name] = metric_rankings

        return rankings

    def _find_best_strategy(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Find best overall strategy based on average rank."""
        rankings = self._generate_rankings(results)

        if not rankings:
            return {"name": None, "avg_rank": None}

        # Calculate average rank for each strategy
        strategy_ranks = {result.strategy_name: [] for result in results}

        for metric_ranks in rankings.values():
            for strategy_name, rank in metric_ranks.items():
                strategy_ranks[strategy_name].append(rank)

        # Compute average ranks
        avg_ranks = {}
        for strategy_name, ranks in strategy_ranks.items():
            if ranks:
                avg_ranks[strategy_name] = np.mean(ranks)

        if not avg_ranks:
            return {"name": None, "avg_rank": None}

        # Best strategy has lowest average rank
        best_name = min(avg_ranks, key=avg_ranks.get)

        return {
            "name": best_name,
            "avg_rank": avg_ranks[best_name],
            "all_ranks": avg_ranks
        }

    def generate_report(
        self,
        comparison: Dict[str, Any],
        format: str = "table"
    ) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison: Comparison data from compare()
            format: Output format ("table", "markdown", or "text")

        Returns:
            Formatted report string

        Example:
            >>> report = comparator.generate_report(comparison, format="markdown")
            >>> print(report)
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("STRATEGY COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Baseline: {comparison['baseline']}")
        lines.append(f"Strategies: {', '.join(comparison['strategies'])}")
        lines.append("")

        # Best strategy
        best = comparison["best_strategy"]
        if best["name"]:
            lines.append(f"Best Overall Strategy: {best['name']} (avg rank: {best['avg_rank']:.2f})")
            lines.append("")

        # Metric comparison table
        lines.append("METRIC COMPARISON")
        lines.append("-" * 80)

        table_data = []
        headers = ["Metric"] + comparison["strategies"]

        for metric_name, metric_data in sorted(comparison["metrics"].items()):
            row = [metric_name]
            for strategy in comparison["strategies"]:
                value = metric_data["values"].get(strategy)
                if value is not None:
                    # Format value
                    if "latency" in metric_name.lower():
                        formatted = f"{value:.2f}ms"
                    elif "cost" in metric_name.lower():
                        formatted = f"${value:.4f}"
                    else:
                        formatted = f"{value:.4f}"

                    # Add improvement indicator
                    improvement = metric_data["improvements"].get(strategy)
                    if improvement is not None and improvement != 0:
                        sign = "+" if improvement > 0 else ""
                        formatted += f" ({sign}{improvement:.1f}%)"

                    # Mark best
                    if strategy == metric_data.get("best"):
                        formatted += " *"

                    row.append(formatted)
                else:
                    row.append("N/A")

            table_data.append(row)

        if format == "markdown":
            lines.append(tabulate(table_data, headers=headers, tablefmt="github"))
        else:
            lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))

        lines.append("")
        lines.append("* Best strategy for metric")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_summary_table(
        self,
        results: List[BenchmarkResult],
        metrics: List[str]
    ) -> str:
        """
        Generate a simple summary table of key metrics.

        Args:
            results: List of benchmark results
            metrics: List of metric names to include

        Returns:
            Formatted table string

        Example:
            >>> table = comparator.generate_summary_table(
            ...     results,
            ...     ["precision@5", "recall@5", "avg_latency_ms"]
            ... )
        """
        headers = ["Strategy"] + metrics
        table_data = []

        for result in results:
            row = [result.strategy_name]
            for metric in metrics:
                value = result.aggregate_metrics.get(metric)
                if value is not None:
                    if "latency" in metric.lower():
                        row.append(f"{value:.2f}ms")
                    elif "cost" in metric.lower():
                        row.append(f"${value:.4f}")
                    else:
                        row.append(f"{value:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)

        return tabulate(table_data, headers=headers, tablefmt="grid")
