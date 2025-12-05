"""JSON exporter for benchmark results."""

import json
from pathlib import Path
from typing import List
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult


class JSONExporter:
    """
    Export benchmark results to JSON format.

    Example:
        >>> exporter = JSONExporter()
        >>> exporter.export(result, "output.json")
    """

    def export(
        self,
        result: BenchmarkResult,
        output_path: str,
        indent: int = 2,
        include_query_details: bool = True
    ) -> None:
        """
        Export benchmark result to JSON.

        Args:
            result: Benchmark result to export
            output_path: Output file path
            indent: JSON indentation level
            include_query_details: Whether to include per-query results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = result.to_dict()
        if not include_query_details:
            data.pop('query_results', None)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)

    def export_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        indent: int = 2
    ) -> None:
        """
        Export comparison of multiple strategies to JSON.

        Args:
            results: List of benchmark results
            output_path: Output file path
            indent: JSON indentation level
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "strategies": [result.strategy_name for result in results],
            "dataset": results[0].dataset_name if results else None,
            "results": [result.to_dict() for result in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)
