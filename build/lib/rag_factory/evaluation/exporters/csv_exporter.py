"""CSV exporter for benchmark results."""

import csv
from pathlib import Path
from typing import List, Optional
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult


class CSVExporter:
    """
    Export benchmark results to CSV format.

    Example:
        >>> exporter = CSVExporter()
        >>> exporter.export(result, "output.csv")
        >>> exporter.export_comparison([result1, result2], "comparison.csv")
    """

    def export(
        self,
        result: BenchmarkResult,
        output_path: str,
        include_query_details: bool = False
    ) -> None:
        """
        Export single benchmark result to CSV.

        Args:
            result: Benchmark result to export
            output_path: Output file path
            include_query_details: Whether to include per-query results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if include_query_details:
            self._export_detailed(result, output_path)
        else:
            self._export_summary(result, output_path)

    def _export_summary(self, result: BenchmarkResult, output_path: Path) -> None:
        """Export summary metrics only."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'Dataset', 'Metric', 'Value'])

            for metric_name, value in sorted(result.aggregate_metrics.items()):
                writer.writerow([
                    result.strategy_name,
                    result.dataset_name,
                    metric_name,
                    f"{value:.6f}"
                ])

    def _export_detailed(self, result: BenchmarkResult, output_path: Path) -> None:
        """Export per-query results."""
        with open(output_path, 'w', newline='') as f:
            # Collect all metric names
            all_metrics = set()
            for qr in result.query_results:
                if "metrics" in qr:
                    all_metrics.update(qr["metrics"].keys())

            fieldnames = ['query_id', 'query', 'latency_ms', 'results_count'] + sorted(all_metrics)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qr in result.query_results:
                if "error" in qr:
                    continue

                row = {
                    'query_id': qr.get('query_id', ''),
                    'query': qr.get('query', '')[:100],  # Truncate long queries
                    'latency_ms': f"{qr.get('latency_ms', 0):.2f}",
                    'results_count': qr.get('results_count', 0)
                }

                # Add metric values
                for metric_name in all_metrics:
                    value = qr.get('metrics', {}).get(metric_name)
                    row[metric_name] = f"{value:.6f}" if value is not None else 'N/A'

                writer.writerow(row)

    def export_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Export comparison of multiple strategies to CSV.

        Args:
            results: List of benchmark results
            output_path: Output file path
            metrics: Optional list of metrics to include (all if None)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all metrics if not specified
        if metrics is None:
            all_metrics = set()
            for result in results:
                all_metrics.update(result.aggregate_metrics.keys())
            metrics = sorted(all_metrics)

        with open(output_path, 'w', newline='') as f:
            fieldnames = ['strategy', 'dataset'] + metrics
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'strategy': result.strategy_name,
                    'dataset': result.dataset_name
                }
                for metric in metrics:
                    value = result.aggregate_metrics.get(metric)
                    row[metric] = f"{value:.6f}" if value is not None else 'N/A'

                writer.writerow(row)
