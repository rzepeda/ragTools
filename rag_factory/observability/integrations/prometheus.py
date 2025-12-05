"""Prometheus metrics exporter for RAG Factory."""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    REGISTRY,
)
from prometheus_client.core import CollectorRegistry
from typing import Optional
import time

from rag_factory.observability.metrics.collector import get_collector


class PrometheusExporter:
    """Export RAG metrics to Prometheus format.

    Provides Prometheus-compatible metrics for monitoring RAG operations
    including queries, latencies, tokens, costs, and errors.

    Example:
        ```python
        from rag_factory.observability.integrations.prometheus import PrometheusExporter

        # Initialize exporter
        exporter = PrometheusExporter()

        # Record a query
        exporter.record_query(
            strategy="vector_search",
            latency_seconds=0.045,
            success=True,
            tokens=150,
            cost=0.001
        )

        # Export metrics
        metrics_output = exporter.export()
        print(metrics_output)
        ```
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus exporter.

        Args:
            registry: Optional custom Prometheus registry
        """
        self.registry = registry or REGISTRY

        # Query metrics
        self.queries_total = Counter(
            'rag_queries_total',
            'Total number of RAG queries',
            ['strategy', 'status'],
            registry=self.registry
        )

        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'RAG query duration in seconds',
            ['strategy'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        # Token metrics
        self.tokens_total = Counter(
            'rag_tokens_total',
            'Total tokens processed',
            ['strategy', 'token_type'],
            registry=self.registry
        )

        # Cost metrics
        self.cost_total = Counter(
            'rag_cost_total_dollars',
            'Total cost in dollars',
            ['strategy'],
            registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['strategy', 'error_type'],
            registry=self.registry
        )

        # Gauge metrics (current state)
        self.active_queries = Gauge(
            'rag_active_queries',
            'Number of currently active queries',
            ['strategy'],
            registry=self.registry
        )

        self.success_rate = Gauge(
            'rag_success_rate',
            'Success rate percentage per strategy',
            ['strategy'],
            registry=self.registry
        )

        # System info
        self.info = Info(
            'rag_factory',
            'RAG Factory information',
            registry=self.registry
        )
        self.info.info({'version': '1.0.0', 'component': 'observability'})

    def record_query(
        self,
        strategy: str,
        latency_seconds: float,
        success: bool = True,
        tokens: int = 0,
        cost: float = 0.0,
        error_type: Optional[str] = None,
    ):
        """Record a query execution in Prometheus metrics.

        Args:
            strategy: Strategy name
            latency_seconds: Query latency in seconds
            success: Whether query succeeded
            tokens: Number of tokens used
            cost: Cost incurred
            error_type: Type of error if query failed
        """
        # Record query count
        status = 'success' if success else 'error'
        self.queries_total.labels(strategy=strategy, status=status).inc()

        # Record latency
        self.query_duration.labels(strategy=strategy).observe(latency_seconds)

        # Record tokens
        if tokens > 0:
            self.tokens_total.labels(
                strategy=strategy,
                token_type='total'
            ).inc(tokens)

        # Record cost
        if cost > 0:
            self.cost_total.labels(strategy=strategy).inc(cost)

        # Record errors
        if not success and error_type:
            self.errors_total.labels(
                strategy=strategy,
                error_type=error_type
            ).inc()

    def update_gauges(self):
        """Update gauge metrics from metrics collector.

        Should be called periodically to sync gauge values.
        """
        collector = get_collector()
        all_metrics = collector.get_metrics()

        for strategy_name, metrics in all_metrics.items():
            # Update success rate
            self.success_rate.labels(strategy=strategy_name).set(
                metrics['success_rate']
            )

    def export(self) -> bytes:
        """Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        self.update_gauges()
        return generate_latest(self.registry)

    def export_text(self) -> str:
        """Export metrics as text string.

        Returns:
            Metrics in Prometheus text format as string
        """
        return self.export().decode('utf-8')


# Global Prometheus exporter instance
_global_exporter: Optional[PrometheusExporter] = None


def get_prometheus_exporter() -> PrometheusExporter:
    """Get or create the global Prometheus exporter instance.

    Returns:
        PrometheusExporter instance
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = PrometheusExporter()
    return _global_exporter


def create_prometheus_endpoint(app):
    """Add Prometheus metrics endpoint to Flask app.

    Args:
        app: Flask application instance

    Example:
        ```python
        from flask import Flask
        from rag_factory.observability.integrations.prometheus import create_prometheus_endpoint

        app = Flask(__name__)
        create_prometheus_endpoint(app)
        ```
    """
    exporter = get_prometheus_exporter()

    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint."""
        return exporter.export(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
