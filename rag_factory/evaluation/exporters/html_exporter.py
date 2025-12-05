"""HTML exporter for benchmark results."""

from pathlib import Path
from typing import List
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult


class HTMLExporter:
    """
    Export benchmark results to HTML format.

    Example:
        >>> exporter = HTMLExporter()
        >>> exporter.export(result, "report.html")
    """

    def export(
        self,
        result: BenchmarkResult,
        output_path: str,
        title: str = "Benchmark Report"
    ) -> None:
        """
        Export benchmark result to HTML.

        Args:
            result: Benchmark result to export
            output_path: Output file path
            title: Report title
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_single_report(result, title)

        with open(output_path, 'w') as f:
            f.write(html)

    def _generate_single_report(self, result: BenchmarkResult, title: str) -> str:
        """Generate HTML report for single strategy."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-value {{
            font-weight: bold;
            color: #27ae60;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }}
        .summary-label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Strategy: {result.strategy_name}</p>
        <p>Dataset: {result.dataset_name}</p>
        <p>Execution Time: {result.execution_time:.2f}s</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-card">
                <div class="summary-label">Total Queries</div>
                <div class="summary-value">{result.metadata.get('total_queries', 0)}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Successful</div>
                <div class="summary-value">{result.metadata.get('successful_queries', 0)}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Failed</div>
                <div class="summary-value">{result.metadata.get('failed_queries', 0)}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Aggregate Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""

        for metric_name, value in sorted(result.aggregate_metrics.items()):
            html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td class="metric-value">{value:.6f}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html

    def export_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        title: str = "Strategy Comparison"
    ) -> None:
        """
        Export comparison of multiple strategies to HTML.

        Args:
            results: List of benchmark results
            output_path: Output file path
            title: Report title
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_comparison_report(results, title)

        with open(output_path, 'w') as f:
            f.write(html)

    def _generate_comparison_report(self, results: List[BenchmarkResult], title: str) -> str:
        """Generate HTML comparison report."""
        # Collect all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.aggregate_metrics.keys())

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Comparing {len(results)} strategies</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Metric</th>
"""

        for result in results:
            html += f"                <th>{result.strategy_name}</th>\n"

        html += """
            </tr>
        </thead>
        <tbody>
"""

        for metric in sorted(all_metrics):
            html += f"            <tr>\n                <td><strong>{metric}</strong></td>\n"

            # Collect values for this metric
            values = []
            for result in results:
                value = result.aggregate_metrics.get(metric)
                values.append(value)

            # Determine best value
            is_lower_better = 'latency' in metric.lower() or 'cost' in metric.lower()
            valid_values = [v for v in values if v is not None]
            best_value = min(valid_values) if (is_lower_better and valid_values) else max(valid_values) if valid_values else None

            # Generate table cells
            for value in values:
                is_best = (value is not None and value == best_value)
                cell_class = ' class="best"' if is_best else ''
                value_str = f"{value:.6f}" if value is not None else "N/A"
                html += f"                <td{cell_class}>{value_str}</td>\n"

            html += "            </tr>\n"

        html += """
        </tbody>
    </table>
</body>
</html>
"""
        return html
