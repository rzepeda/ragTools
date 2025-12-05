"""Unit tests for metrics collector."""

import pytest
import threading
from datetime import datetime, timedelta

from rag_factory.observability.metrics.collector import (
    MetricsCollector,
    PerformanceMetrics,
    MetricPoint,
    get_collector,
)


@pytest.fixture
def collector():
    """Create a MetricsCollector instance for testing."""
    return MetricsCollector()


@pytest.fixture
def reset_global_collector():
    """Reset global collector after each test."""
    yield
    import rag_factory.observability.metrics.collector as collector_module
    collector_module._global_collector = None


class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        timestamp = datetime.now()
        point = MetricPoint(
            timestamp=timestamp,
            value=45.2,
            labels={"strategy": "test", "success": "true"},
        )

        assert point.timestamp == timestamp
        assert point.value == 45.2
        assert point.labels["strategy"] == "test"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(strategy_name="test_strategy")

        assert metrics.strategy_name == "test_strategy"
        assert metrics.total_queries == 0
        assert metrics.successful_queries == 0
        assert metrics.failed_queries == 0
        assert metrics.total_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = PerformanceMetrics(strategy_name="test")
        metrics.total_queries = 5
        metrics.total_latency_ms = 250.0

        assert metrics.avg_latency_ms == 50.0

    def test_avg_latency_no_queries(self):
        """Test average latency with no queries."""
        metrics = PerformanceMetrics(strategy_name="test")
        assert metrics.avg_latency_ms == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PerformanceMetrics(strategy_name="test")
        metrics.total_queries = 10
        metrics.successful_queries = 7

        assert metrics.success_rate == 70.0

    def test_success_rate_no_queries(self):
        """Test success rate with no queries."""
        metrics = PerformanceMetrics(strategy_name="test")
        assert metrics.success_rate == 0.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = PerformanceMetrics(strategy_name="test")
        metrics.total_queries = 10
        metrics.failed_queries = 3

        assert metrics.failure_rate == 30.0

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = PerformanceMetrics(strategy_name="test")

        # Add latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for latency in latencies:
            metrics.latencies.append(latency)

        # Check percentiles (approximate)
        assert 40 <= metrics.p50_latency <= 60
        assert 90 <= metrics.p95_latency <= 100
        assert 95 <= metrics.p99_latency <= 100

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        metrics = PerformanceMetrics(strategy_name="test")
        metrics.latencies.append(50.0)

        assert metrics.p50_latency == 50.0
        assert metrics.p95_latency == 50.0
        assert metrics.p99_latency == 50.0

    def test_percentile_no_values(self):
        """Test percentile with no values."""
        metrics = PerformanceMetrics(strategy_name="test")

        assert metrics.p50_latency == 0.0
        assert metrics.p95_latency == 0.0
        assert metrics.p99_latency == 0.0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_collector_initialization(self, collector):
        """Test collector initializes correctly."""
        assert len(collector._metrics) == 0
        assert collector._start_time > 0

    def test_record_successful_query(self, collector):
        """Test recording a successful query."""
        collector.record_query(
            strategy="vector_search",
            latency_ms=45.2,
            tokens=150,
            cost=0.001,
            success=True,
        )

        metrics = collector.get_metrics("vector_search")

        assert metrics["total_queries"] == 1
        assert metrics["successful_queries"] == 1
        assert metrics["failed_queries"] == 0
        assert metrics["avg_latency_ms"] == 45.2
        assert metrics["total_tokens"] == 150
        assert metrics["total_cost"] == 0.001

    def test_record_failed_query(self, collector):
        """Test recording a failed query."""
        collector.record_query(
            strategy="vector_search",
            latency_ms=30.0,
            success=False,
            error="Connection timeout",
        )

        metrics = collector.get_metrics("vector_search")

        assert metrics["total_queries"] == 1
        assert metrics["successful_queries"] == 0
        assert metrics["failed_queries"] == 1
        assert metrics["error_count"] == 1

    def test_multiple_queries(self, collector):
        """Test recording multiple queries."""
        for i in range(10):
            collector.record_query(
                strategy="vector_search",
                latency_ms=40.0 + i,
                tokens=100,
                cost=0.001,
                success=True,
            )

        metrics = collector.get_metrics("vector_search")

        assert metrics["total_queries"] == 10
        assert metrics["successful_queries"] == 10
        assert metrics["total_tokens"] == 1000
        assert metrics["total_cost"] == 0.01

    def test_success_rate_calculation(self, collector):
        """Test success rate calculation."""
        # 7 successful, 3 failed
        for i in range(10):
            collector.record_query(
                strategy="test",
                latency_ms=50.0,
                success=(i < 7),
            )

        metrics = collector.get_metrics("test")

        assert metrics["success_rate"] == 70.0
        assert metrics["failure_rate"] == 30.0

    def test_multiple_strategies(self, collector):
        """Test metrics for multiple strategies."""
        collector.record_query(
            strategy="strategy_a", latency_ms=10, success=True
        )
        collector.record_query(
            strategy="strategy_b", latency_ms=20, success=True
        )
        collector.record_query(
            strategy="strategy_a", latency_ms=15, success=True
        )

        all_metrics = collector.get_metrics()

        assert len(all_metrics) == 2
        assert "strategy_a" in all_metrics
        assert "strategy_b" in all_metrics
        assert all_metrics["strategy_a"]["total_queries"] == 2
        assert all_metrics["strategy_b"]["total_queries"] == 1

    def test_get_metrics_nonexistent_strategy(self, collector):
        """Test getting metrics for non-existent strategy."""
        metrics = collector.get_metrics("nonexistent")
        assert metrics == {}

    def test_get_summary(self, collector):
        """Test overall summary calculation."""
        collector.record_query(
            strategy="a", latency_ms=10, cost=0.001, success=True
        )
        collector.record_query(
            strategy="b", latency_ms=20, cost=0.002, success=True
        )
        collector.record_query(
            strategy="a", latency_ms=15, cost=0.001, success=False
        )

        summary = collector.get_summary()

        assert summary["total_queries"] == 3
        assert summary["successful_queries"] == 2
        assert summary["failed_queries"] == 1
        assert summary["overall_success_rate"] == pytest.approx(66.67, rel=0.1)
        assert summary["total_cost"] == 0.004
        assert summary["strategies_count"] == 2
        assert "uptime_seconds" in summary
        assert "queries_per_second" in summary

    def test_get_strategy_names(self, collector):
        """Test getting list of strategy names."""
        collector.record_query(strategy="strategy_a", latency_ms=10, success=True)
        collector.record_query(strategy="strategy_b", latency_ms=20, success=True)

        names = collector.get_strategy_names()

        assert len(names) == 2
        assert "strategy_a" in names
        assert "strategy_b" in names

    def test_reset_specific_strategy(self, collector):
        """Test resetting metrics for specific strategy."""
        collector.record_query(strategy="test_a", latency_ms=10, success=True)
        collector.record_query(strategy="test_b", latency_ms=20, success=True)

        collector.reset_metrics("test_a")

        metrics_a = collector.get_metrics("test_a")
        metrics_b = collector.get_metrics("test_b")

        assert metrics_a == {}
        assert metrics_b["total_queries"] == 1

    def test_reset_all_metrics(self, collector):
        """Test resetting all metrics."""
        collector.record_query(strategy="test", latency_ms=10, success=True)

        collector.reset_metrics()

        all_metrics = collector.get_metrics()
        assert len(all_metrics) == 0

    def test_time_series_recording(self, collector):
        """Test time-series data recording."""
        collector.record_query(
            strategy="test", latency_ms=50, tokens=100, cost=0.001, success=True
        )

        # Get time series for latency
        latency_series = collector.get_time_series("test_latency", duration_minutes=60)

        assert len(latency_series) > 0
        assert latency_series[0].value == 50

    def test_time_series_time_window(self, collector):
        """Test time-series data filtering by time window."""
        # Record a query
        collector.record_query(strategy="test", latency_ms=50, success=True)

        # Get time series with 0 minute window (should be empty)
        series = collector.get_time_series("test_latency", duration_minutes=0)
        assert len(series) == 0

        # Get time series with 60 minute window (should have data)
        series = collector.get_time_series("test_latency", duration_minutes=60)
        assert len(series) > 0

    def test_time_series_nonexistent_metric(self, collector):
        """Test getting time series for non-existent metric."""
        series = collector.get_time_series("nonexistent", duration_minutes=60)
        assert series == []

    def test_thread_safety(self, collector):
        """Test thread-safe operations."""

        def record_queries():
            for _ in range(100):
                collector.record_query(
                    strategy="test",
                    latency_ms=50,
                    success=True,
                )

        threads = [threading.Thread(target=record_queries) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = collector.get_metrics("test")
        assert metrics["total_queries"] == 1000

    def test_error_tracking(self, collector):
        """Test error tracking and limiting."""
        # Record more than 100 errors
        for i in range(150):
            collector.record_query(
                strategy="test",
                latency_ms=50,
                success=False,
                error=f"Error {i}",
            )

        metrics = collector.get_metrics("test")

        # Should keep only last 100 errors
        assert metrics["error_count"] == 100
        assert len(metrics["recent_errors"]) == 5  # Last 5 shown in metrics


class TestGetCollector:
    """Tests for get_collector function."""

    def test_get_collector_singleton(self, reset_global_collector):
        """Test that get_collector returns singleton instance."""
        collector1 = get_collector()
        collector2 = get_collector()

        assert collector1 is collector2

    def test_get_collector_persistence(self, reset_global_collector):
        """Test that collector persists data across calls."""
        collector1 = get_collector()
        collector1.record_query(strategy="test", latency_ms=50, success=True)

        collector2 = get_collector()
        metrics = collector2.get_metrics("test")

        assert metrics["total_queries"] == 1
