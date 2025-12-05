"""Integration tests for monitoring system."""

import pytest
import time

from rag_factory.observability.logging.logger import RAGLogger
from rag_factory.observability.metrics.collector import MetricsCollector
from rag_factory.observability.metrics.cost import CostCalculator
from rag_factory.observability.metrics.performance import PerformanceMonitor


@pytest.fixture
def logger():
    """Create logger for testing."""
    return RAGLogger()


@pytest.fixture
def collector():
    """Create metrics collector for testing."""
    return MetricsCollector()


@pytest.fixture
def cost_calculator():
    """Create cost calculator for testing."""
    return CostCalculator()


@pytest.fixture
def performance_monitor():
    """Create performance monitor for testing."""
    return PerformanceMonitor()


@pytest.mark.integration
class TestCompleteQueryLogging:
    """Integration tests for complete query execution logging."""

    def test_complete_query_workflow(self, logger, collector, cost_calculator):
        """Test complete query execution with logging and metrics."""
        strategy = "vector_search"
        query = "test query"
        model = "gpt-3.5-turbo"

        # Simulate query execution with logging
        start_time = time.time()

        with logger.operation("retrieve", strategy=strategy, query=query) as ctx:
            # Simulate retrieval
            time.sleep(0.05)  # 50ms
            results_count = 5
            tokens = 150
            ctx.metadata["results_count"] = results_count
            ctx.metadata["tokens"] = tokens

        latency_ms = ctx.elapsed_ms()

        # Calculate cost
        cost = cost_calculator.calculate_cost(model, input_tokens=tokens, output_tokens=0)

        # Record metrics
        collector.record_query(
            strategy=strategy,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            success=True,
        )

        # Verify metrics
        metrics = collector.get_metrics(strategy)

        assert metrics["total_queries"] == 1
        assert metrics["successful_queries"] == 1
        assert metrics["avg_latency_ms"] >= 50
        assert metrics["total_tokens"] == tokens
        assert metrics["total_cost"] > 0

    def test_failed_query_workflow(self, logger, collector):
        """Test failed query execution with error logging."""
        strategy = "test_strategy"

        # Simulate failed query
        try:
            with logger.operation("retrieve", strategy=strategy) as ctx:
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Record failed query
        collector.record_query(
            strategy=strategy,
            latency_ms=ctx.elapsed_ms(),
            success=False,
            error="ValueError: Simulated error",
        )

        # Verify metrics
        metrics = collector.get_metrics(strategy)

        assert metrics["total_queries"] == 1
        assert metrics["failed_queries"] == 1
        assert metrics["error_count"] == 1

    def test_multiple_strategy_workflow(self, logger, collector):
        """Test logging multiple strategies concurrently."""
        strategies = ["vector_search", "bm25", "hybrid"]

        for strategy in strategies:
            for i in range(5):
                with logger.operation("retrieve", strategy=strategy, query=f"query {i}") as ctx:
                    time.sleep(0.01)

                collector.record_query(
                    strategy=strategy,
                    latency_ms=ctx.elapsed_ms(),
                    success=True,
                )

        # Verify all strategies tracked
        all_metrics = collector.get_metrics()

        assert len(all_metrics) == 3
        for strategy in strategies:
            assert strategy in all_metrics
            assert all_metrics[strategy]["total_queries"] == 5


@pytest.mark.integration
class TestPerformanceMonitoring:
    """Integration tests for performance monitoring."""

    def test_performance_tracking(self, performance_monitor):
        """Test performance tracking during operations."""
        with performance_monitor.track("embedding_generation"):
            # Simulate work
            time.sleep(0.02)

        stats = performance_monitor.get_stats("embedding_generation")

        assert stats["executions"] == 1
        assert stats["avg_duration_ms"] >= 20
        assert "avg_cpu_percent" in stats
        assert "avg_memory_percent" in stats

    def test_multiple_operations_tracking(self, performance_monitor):
        """Test tracking multiple operations."""
        operations = ["retrieve", "rerank", "generate"]

        for operation in operations:
            with performance_monitor.track(operation):
                time.sleep(0.01)

        all_stats = performance_monitor.get_all_stats()

        assert len(all_stats) == 3
        for operation in operations:
            assert operation in all_stats
            assert all_stats[operation]["executions"] == 1


@pytest.mark.integration
class TestDashboardAPI:
    """Integration tests for dashboard API."""

    def test_dashboard_endpoints(self, collector):
        """Test dashboard API endpoints."""
        from rag_factory.observability.monitoring.api import app

        # Record some test data
        collector.record_query(strategy="test", latency_ms=50, success=True)

        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/api/health')
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "healthy"

            # Test metrics endpoint
            response = client.get('/api/metrics')
            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True

            # Test summary endpoint
            response = client.get('/api/metrics/summary')
            assert response.status_code == 200
            data = response.get_json()
            assert "total_queries" in data["data"]

            # Test strategies endpoint
            response = client.get('/api/strategies')
            assert response.status_code == 200

            # Test system endpoint
            response = client.get('/api/system')
            assert response.status_code == 200
            data = response.get_json()
            assert "cpu_percent" in data["data"]

    def test_dashboard_error_handling(self):
        """Test dashboard API error handling."""
        from rag_factory.observability.monitoring.api import app

        with app.test_client() as client:
            # Test 404
            response = client.get('/api/nonexistent')
            assert response.status_code == 404

            # Test time series without metric parameter
            response = client.get('/api/metrics/timeseries')
            assert response.status_code == 400


@pytest.mark.integration
class TestEndToEndMonitoring:
    """End-to-end integration tests."""

    def test_complete_monitoring_stack(
        self, logger, collector, cost_calculator, performance_monitor
    ):
        """Test complete monitoring stack integration."""
        strategy = "complete_test"
        query = "integration test query"
        model = "gpt-3.5-turbo"

        # Use all monitoring components together
        with performance_monitor.track("complete_query"):
            with logger.operation("retrieve", strategy=strategy, query=query) as ctx:
                # Simulate retrieval
                time.sleep(0.03)
                results_count = 10
                tokens = 200
                ctx.metadata["results_count"] = results_count
                ctx.metadata["tokens"] = tokens

            latency_ms = ctx.elapsed_ms()

            # Calculate cost
            cost = cost_calculator.calculate_cost(model, input_tokens=tokens, output_tokens=50)

            # Record metrics
            collector.record_query(
                strategy=strategy,
                latency_ms=latency_ms,
                tokens=tokens,
                cost=cost,
                success=True,
            )

        # Verify all components tracked data
        metrics = collector.get_metrics(strategy)
        perf_stats = performance_monitor.get_stats("complete_query")
        summary = collector.get_summary()

        assert metrics["total_queries"] == 1
        assert metrics["total_cost"] > 0
        assert perf_stats["executions"] == 1
        assert summary["total_queries"] >= 1

    def test_high_volume_simulation(self, logger, collector):
        """Test monitoring system under high volume."""
        num_queries = 100
        strategies = ["strategy_a", "strategy_b", "strategy_c"]

        for i in range(num_queries):
            strategy = strategies[i % len(strategies)]

            with logger.operation("retrieve", strategy=strategy, query=f"query {i}") as ctx:
                time.sleep(0.001)  # 1ms

            collector.record_query(
                strategy=strategy,
                latency_ms=ctx.elapsed_ms(),
                tokens=100,
                cost=0.0001,
                success=True,
            )

        # Verify metrics
        summary = collector.get_summary()

        assert summary["total_queries"] == num_queries
        assert summary["strategies_count"] == len(strategies)
        assert summary["queries_per_second"] > 0

    def test_error_rate_monitoring(self, logger, collector):
        """Test monitoring error rates."""
        strategy = "error_test"
        total = 20
        failures = 5

        for i in range(total):
            success = i >= failures  # First 5 fail, rest succeed

            try:
                with logger.operation("retrieve", strategy=strategy) as ctx:
                    if not success:
                        raise ValueError(f"Error {i}")
                    time.sleep(0.001)
            except ValueError:
                pass

            collector.record_query(
                strategy=strategy,
                latency_ms=ctx.elapsed_ms(),
                success=success,
                error=f"ValueError: Error {i}" if not success else None,
            )

        # Verify error rate
        metrics = collector.get_metrics(strategy)

        assert metrics["total_queries"] == total
        assert metrics["failed_queries"] == failures
        assert metrics["success_rate"] == 75.0  # 15/20 = 75%


@pytest.mark.integration
class TestPrometheusIntegration:
    """Integration tests for Prometheus exporter."""

    def test_prometheus_metrics_export(self, collector):
        """Test Prometheus metrics export."""
        from rag_factory.observability.integrations.prometheus import PrometheusExporter

        exporter = PrometheusExporter()

        # Record some queries
        collector.record_query(strategy="test", latency_ms=50, tokens=100, cost=0.001, success=True)
        collector.record_query(strategy="test", latency_ms=75, tokens=150, cost=0.0015, success=False)

        # Record to Prometheus exporter
        exporter.record_query(strategy="test", latency_seconds=0.05, tokens=100, cost=0.001, success=True)
        exporter.record_query(
            strategy="test",
            latency_seconds=0.075,
            tokens=150,
            cost=0.0015,
            success=False,
            error_type="ValueError"
        )

        # Export metrics
        metrics_text = exporter.export_text()

        assert "rag_queries_total" in metrics_text
        assert "rag_query_duration_seconds" in metrics_text
        assert "rag_tokens_total" in metrics_text
        assert "rag_cost_total_dollars" in metrics_text
