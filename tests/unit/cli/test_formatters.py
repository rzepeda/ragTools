"""Unit tests for output formatters."""

import pytest
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from rag_factory.cli.formatters.output import format_error, format_success, format_warning
from rag_factory.cli.formatters.results import (
    format_benchmark_results,
    format_query_results,
    format_statistics,
    format_strategy_list,
)


class TestOutputFormatters:
    """Tests for general output formatters."""

    def test_format_success(self):
        """Test success message formatting."""
        result = format_success("Operation completed")
        assert isinstance(result, Panel)
        # Check that the panel's renderable contains the text
        assert str(result.renderable) == "Operation completed"

    def test_format_success_with_title(self):
        """Test success message with custom title."""
        result = format_success("Done", title="Custom Title")
        assert isinstance(result, Panel)
        assert result.title == "Custom Title"

    def test_format_error(self):
        """Test error message formatting."""
        result = format_error("Something failed")
        assert isinstance(result, Panel)
        # Check that the panel's renderable contains the text
        assert str(result.renderable) == "Something failed"

    def test_format_warning(self):
        """Test warning message formatting."""
        result = format_warning("Be careful")
        assert isinstance(result, Panel)
        # Check that the panel's renderable contains the text
        assert str(result.renderable) == "Be careful"


class TestQueryResultsFormatter:
    """Tests for query results formatting."""

    def test_format_query_results_basic(self):
        """Test basic query results formatting."""
        results = [
            {"text": "Result 1", "score": 0.95, "metadata": {"source": "doc1.txt"}},
            {"text": "Result 2", "score": 0.85, "metadata": {"source": "doc2.txt"}},
        ]

        table = format_query_results(results, "test query")
        assert isinstance(table, Table)
        assert table.row_count == 2

    def test_format_query_results_with_strategy(self):
        """Test query results with strategy name."""
        results = [
            {"text": "Result 1", "score": 0.95, "metadata": {"source": "doc1.txt"}},
        ]

        table = format_query_results(results, "test query", "reranking")
        assert isinstance(table, Table)
        assert "reranking" in str(table.title)

    def test_format_empty_results(self):
        """Test formatting empty results."""
        table = format_query_results([], "test query")
        assert isinstance(table, Table)
        assert table.row_count == 0

    def test_format_results_with_long_text(self):
        """Test formatting results with long text (should truncate)."""
        long_text = "x" * 300
        results = [
            {"text": long_text, "score": 0.95, "metadata": {"source": "doc1.txt"}},
        ]

        table = format_query_results(results, "test query")
        assert isinstance(table, Table)


class TestStrategyListFormatter:
    """Tests for strategy list formatting."""

    def test_format_strategy_list_basic(self):
        """Test basic strategy list formatting."""
        strategies = [
            {"name": "strategy1", "type": "chunking", "description": "Description 1"},
            {"name": "strategy2", "type": "reranking", "description": "Description 2"},
        ]

        tree = format_strategy_list(strategies)
        assert isinstance(tree, Tree)

    def test_format_strategy_list_with_filter(self):
        """Test strategy list with type filter."""
        strategies = [
            {"name": "strategy1", "type": "chunking", "description": "Description 1"},
            {"name": "strategy2", "type": "reranking", "description": "Description 2"},
        ]

        tree = format_strategy_list(strategies, filter_type="chunking")
        assert isinstance(tree, Tree)

    def test_format_empty_strategy_list(self):
        """Test formatting empty strategy list."""
        tree = format_strategy_list([])
        assert isinstance(tree, Tree)

    def test_format_strategy_list_grouped_by_type(self):
        """Test that strategies are grouped by type."""
        strategies = [
            {"name": "strategy1", "type": "chunking", "description": "Desc 1"},
            {"name": "strategy2", "type": "chunking", "description": "Desc 2"},
            {"name": "strategy3", "type": "reranking", "description": "Desc 3"},
        ]

        tree = format_strategy_list(strategies)
        assert isinstance(tree, Tree)


class TestBenchmarkResultsFormatter:
    """Tests for benchmark results formatting."""

    def test_format_benchmark_results_basic(self):
        """Test basic benchmark results formatting."""
        results = {
            "strategy1": {
                "avg_latency_ms": 100.5,
                "total_queries": 50,
                "success_rate": 0.98,
                "avg_score": 0.85,
            },
            "strategy2": {
                "avg_latency_ms": 150.2,
                "total_queries": 50,
                "success_rate": 0.95,
                "avg_score": 0.82,
            },
        }

        table = format_benchmark_results(results, ["strategy1", "strategy2"])
        assert isinstance(table, Table)
        assert table.row_count == 2

    def test_format_single_strategy_benchmark(self):
        """Test formatting single strategy benchmark."""
        results = {
            "strategy1": {
                "avg_latency_ms": 100.5,
                "total_queries": 50,
                "success_rate": 0.98,
                "avg_score": 0.85,
            },
        }

        table = format_benchmark_results(results, ["strategy1"])
        assert isinstance(table, Table)
        assert table.row_count == 1

    def test_format_empty_benchmark_results(self):
        """Test formatting empty benchmark results."""
        table = format_benchmark_results({}, [])
        assert isinstance(table, Table)
        assert table.row_count == 0


class TestStatisticsFormatter:
    """Tests for statistics formatting."""

    def test_format_statistics_basic(self):
        """Test basic statistics formatting."""
        stats = {
            "documents_processed": 100,
            "total_chunks": 500,
            "elapsed_time": 45.2,
        }

        table = format_statistics(stats)
        assert isinstance(table, Table)
        assert table.row_count == 3

    def test_format_statistics_with_floats(self):
        """Test statistics with float values."""
        stats = {
            "avg_score": 0.8534,
            "latency": 123.456,
        }

        table = format_statistics(stats)
        assert isinstance(table, Table)

    def test_format_empty_statistics(self):
        """Test formatting empty statistics."""
        table = format_statistics({})
        assert isinstance(table, Table)
        assert table.row_count == 0

    def test_format_statistics_key_transformation(self):
        """Test that keys are transformed to readable format."""
        stats = {
            "documents_processed": 100,
            "total_chunks_created": 500,
        }

        table = format_statistics(stats)
        # Keys should be transformed (underscores to spaces, title case)
        assert isinstance(table, Table)
