"""
Unit tests for export functionality.

Tests CSV and JSON exporters for benchmark results.
"""

import pytest
import json
import csv
from pathlib import Path
from typing import Dict, Any
from rag_factory.evaluation.benchmarks.runner import BenchmarkResult
from rag_factory.evaluation.exporters.csv_exporter import CSVExporter
from rag_factory.evaluation.exporters.json_exporter import JSONExporter


@pytest.fixture
def sample_benchmark_result() -> BenchmarkResult:
    """Create a sample benchmark result."""
    return BenchmarkResult(
        strategy_name="TestStrategy",
        dataset_name="test_dataset",
        query_results=[
            {
                "query_id": "q1",
                "query": "What is machine learning?",
                "latency_ms": 150.5,
                "results_count": 5,
                "retrieved_ids": ["doc1", "doc2", "doc3"],
                "metrics": {
                    "precision@5": 0.8,
                    "recall@5": 0.6,
                    "ndcg@5": 0.75
                }
            },
            {
                "query_id": "q2",
                "query": "What is AI?",
                "latency_ms": 120.3,
                "results_count": 5,
                "retrieved_ids": ["doc2", "doc4", "doc5"],
                "metrics": {
                    "precision@5": 0.9,
                    "recall@5": 0.7,
                    "ndcg@5": 0.85
                }
            }
        ],
        aggregate_metrics={
            "precision@5": 0.85,
            "recall@5": 0.65,
            "ndcg@5": 0.80,
            "avg_latency_ms": 135.4
        },
        execution_time=2.5,
        metadata={"total_queries": 2, "successful_queries": 2}
    )


@pytest.fixture
def multiple_benchmark_results() -> list[BenchmarkResult]:
    """Create multiple benchmark results for comparison."""
    return [
        BenchmarkResult(
            strategy_name="Strategy1",
            dataset_name="test_dataset",
            query_results=[],
            aggregate_metrics={
                "precision@5": 0.85,
                "recall@5": 0.65,
                "ndcg@5": 0.80
            },
            execution_time=2.5
        ),
        BenchmarkResult(
            strategy_name="Strategy2",
            dataset_name="test_dataset",
            query_results=[],
            aggregate_metrics={
                "precision@5": 0.90,
                "recall@5": 0.70,
                "ndcg@5": 0.85
            },
            execution_time=3.0
        )
    ]


class TestCSVExporter:
    """Tests for CSVExporter class."""

    def test_csv_export_summary(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test CSV export with summary mode."""
        exporter = CSVExporter()
        output_path = tmp_path / "summary.csv"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=False
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 4  # 4 aggregate metrics
        assert all(row['Strategy'] == 'TestStrategy' for row in rows)
        assert all(row['Dataset'] == 'test_dataset' for row in rows)

    def test_csv_export_detailed(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test CSV export with detailed query results."""
        exporter = CSVExporter()
        output_path = tmp_path / "detailed.csv"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=True
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # 2 queries
        assert rows[0]['query_id'] == 'q1'
        assert rows[1]['query_id'] == 'q2'
        assert 'precision@5' in rows[0]
        assert float(rows[0]['precision@5']) == 0.8

    def test_csv_export_comparison(
        self,
        multiple_benchmark_results: list[BenchmarkResult],
        tmp_path: Path
    ) -> None:
        """Test CSV export for strategy comparison."""
        exporter = CSVExporter()
        output_path = tmp_path / "comparison.csv"
        
        exporter.export_comparison(
            results=multiple_benchmark_results,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # 2 strategies
        assert rows[0]['strategy'] == 'Strategy1'
        assert rows[1]['strategy'] == 'Strategy2'
        assert 'precision@5' in rows[0]
        assert float(rows[0]['precision@5']) == 0.85

    def test_csv_export_comparison_custom_metrics(
        self,
        multiple_benchmark_results: list[BenchmarkResult],
        tmp_path: Path
    ) -> None:
        """Test CSV export with custom metric selection."""
        exporter = CSVExporter()
        output_path = tmp_path / "comparison_custom.csv"
        
        exporter.export_comparison(
            results=multiple_benchmark_results,
            output_path=str(output_path),
            metrics=["precision@5", "recall@5"]
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        assert 'precision@5' in fieldnames
        assert 'recall@5' in fieldnames
        assert 'ndcg@5' not in fieldnames  # Should be excluded

    def test_csv_export_creates_directories(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test that CSV export creates parent directories."""
        exporter = CSVExporter()
        output_path = tmp_path / "subdir" / "nested" / "output.csv"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=False
        )
        
        assert output_path.exists()
        assert output_path.parent.exists()


class TestJSONExporter:
    """Tests for JSONExporter class."""

    def test_json_export_with_details(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test JSON export with query details."""
        exporter = JSONExporter()
        output_path = tmp_path / "result.json"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=True
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data['strategy_name'] == 'TestStrategy'
        assert data['dataset_name'] == 'test_dataset'
        assert 'query_results' in data
        assert len(data['query_results']) == 2
        assert data['aggregate_metrics']['precision@5'] == 0.85

    def test_json_export_without_details(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test JSON export without query details."""
        exporter = JSONExporter()
        output_path = tmp_path / "result_summary.json"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=False
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'query_results' not in data
        assert 'aggregate_metrics' in data
        assert data['strategy_name'] == 'TestStrategy'

    def test_json_export_comparison(
        self,
        multiple_benchmark_results: list[BenchmarkResult],
        tmp_path: Path
    ) -> None:
        """Test JSON export for strategy comparison."""
        exporter = JSONExporter()
        output_path = tmp_path / "comparison.json"
        
        exporter.export_comparison(
            results=multiple_benchmark_results,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'strategies' in data
        assert len(data['strategies']) == 2
        assert data['strategies'] == ['Strategy1', 'Strategy2']
        assert 'results' in data
        assert len(data['results']) == 2

    def test_json_export_indentation(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test JSON export with custom indentation."""
        exporter = JSONExporter()
        output_path = tmp_path / "result_indent.json"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            indent=4
        )
        
        assert output_path.exists()
        
        # Verify indentation by checking file content
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Should have 4-space indentation
        assert '    "strategy_name"' in content

    def test_json_export_creates_directories(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test that JSON export creates parent directories."""
        exporter = JSONExporter()
        output_path = tmp_path / "subdir" / "nested" / "output.json"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_export_file_content_validity(
        self,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path
    ) -> None:
        """Test that exported JSON content is valid and complete."""
        exporter = JSONExporter()
        output_path = tmp_path / "valid.json"
        
        exporter.export(
            result=sample_benchmark_result,
            output_path=str(output_path),
            include_query_details=True
        )
        
        # Load and validate structure
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Verify all expected fields
        assert 'strategy_name' in data
        assert 'dataset_name' in data
        assert 'query_results' in data
        assert 'aggregate_metrics' in data
        assert 'execution_time' in data
        assert 'metadata' in data
        
        # Verify query results structure
        for query_result in data['query_results']:
            assert 'query_id' in query_result
            assert 'query' in query_result
            assert 'metrics' in query_result

    def test_csv_field_selection(
        self,
        multiple_benchmark_results: list[BenchmarkResult],
        tmp_path: Path
    ) -> None:
        """Test CSV export with specific field selection."""
        exporter = CSVExporter()
        output_path = tmp_path / "selected_fields.csv"
        
        # Export only precision and recall
        exporter.export_comparison(
            results=multiple_benchmark_results,
            output_path=str(output_path),
            metrics=["precision@5"]
        )
        
        # Read and verify
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
        
        assert 'precision@5' in fieldnames
        assert 'recall@5' not in fieldnames
        assert 'ndcg@5' not in fieldnames
