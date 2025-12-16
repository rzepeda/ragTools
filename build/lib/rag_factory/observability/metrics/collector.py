"""Metrics collection for RAG operations."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
from threading import Lock
import statistics


@dataclass
class MetricPoint:
    """Single metric data point for time-series tracking.

    Attributes:
        timestamp: When the metric was recorded
        value: Metric value
        labels: Additional labels for the metric
    """

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a RAG strategy.

    Tracks comprehensive performance data including:
    - Query counts and success rates
    - Latency statistics (average, percentiles)
    - Token usage and costs
    - Error tracking

    Attributes:
        strategy_name: Name of the strategy
        total_queries: Total number of queries executed
        successful_queries: Number of successful queries
        failed_queries: Number of failed queries
        total_latency_ms: Cumulative latency in milliseconds
        total_tokens: Total tokens used (for LLM-based strategies)
        total_cost: Total cost incurred (for API-based strategies)
        latencies: Recent latencies for percentile calculations
        errors: Recent error messages
    """

    strategy_name: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    errors: List[str] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds.

        Returns:
            Average latency or 0.0 if no queries
        """
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate between 0-100 or 0.0 if no queries
        """
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage.

        Returns:
            Failure rate between 0-100 or 0.0 if no queries
        """
        if self.total_queries == 0:
            return 0.0
        return (self.failed_queries / self.total_queries) * 100

    @property
    def p50_latency(self) -> float:
        """Calculate 50th percentile (median) latency.

        Returns:
            50th percentile latency or 0.0 if insufficient data
        """
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency.

        Returns:
            95th percentile latency or 0.0 if insufficient data
        """
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency.

        Returns:
            99th percentile latency or 0.0 if insufficient data
        """
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """Calculate percentile from latency deque.

        Args:
            p: Percentile to calculate (1-100)

        Returns:
            Percentile value or 0.0 if insufficient data
        """
        if not self.latencies:
            return 0.0

        latencies_list = sorted(list(self.latencies))

        if len(latencies_list) == 1:
            return latencies_list[0]

        # Calculate percentile index
        index = (p / 100) * (len(latencies_list) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(latencies_list) - 1)

        # Linear interpolation
        weight = index - lower_index
        return latencies_list[lower_index] * (1 - weight) + latencies_list[
            upper_index
        ] * weight


class MetricsCollector:
    """Centralized metrics collection for RAG operations.

    Thread-safe collector that tracks performance metrics, costs,
    and query analytics across multiple strategies.

    Features:
    - Per-strategy performance metrics
    - Cost tracking with token usage
    - Query analytics with percentiles
    - Time-series data collection
    - Thread-safe operations
    - In-memory storage with configurable retention

    Example:
        ```python
        collector = MetricsCollector()

        # Record a successful query
        collector.record_query(
            strategy="vector_search",
            latency_ms=45.2,
            tokens=150,
            cost=0.001,
            success=True
        )

        # Get metrics for a strategy
        metrics = collector.get_metrics("vector_search")
        print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"Success rate: {metrics['success_rate']:.1f}%")

        # Get overall summary
        summary = collector.get_summary()
        print(f"Total queries: {summary['total_queries']}")
        print(f"Total cost: ${summary['total_cost']:.4f}")
        ```
    """

    def __init__(self, max_time_series_points: int = 10000):
        """Initialize metrics collector.

        Args:
            max_time_series_points: Maximum time-series points to keep per metric
        """
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = Lock()
        self._time_series: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_time_series_points)
        )
        self._start_time = time.time()

    def record_query(
        self,
        strategy: str,
        latency_ms: float,
        tokens: int = 0,
        cost: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        **metadata,
    ):
        """Record a query execution with metrics.

        Args:
            strategy: Strategy name
            latency_ms: Query latency in milliseconds
            tokens: Number of tokens used (for LLM operations)
            cost: Cost incurred (for API operations)
            success: Whether query succeeded
            error: Error message if query failed
            **metadata: Additional metadata to track
        """
        with self._lock:
            # Get or create metrics for strategy
            if strategy not in self._metrics:
                self._metrics[strategy] = PerformanceMetrics(
                    strategy_name=strategy
                )

            metrics = self._metrics[strategy]
            metrics.total_queries += 1

            if success:
                metrics.successful_queries += 1
            else:
                metrics.failed_queries += 1
                if error:
                    # Keep last 100 errors
                    if len(metrics.errors) >= 100:
                        metrics.errors.pop(0)
                    metrics.errors.append(error)

            metrics.total_latency_ms += latency_ms
            metrics.latencies.append(latency_ms)
            metrics.total_tokens += tokens
            metrics.total_cost += cost

            # Record time-series points
            timestamp = datetime.now()
            self._time_series[f"{strategy}_latency"].append(
                MetricPoint(
                    timestamp=timestamp,
                    value=latency_ms,
                    labels={"strategy": strategy, "success": str(success)},
                )
            )

            if tokens > 0:
                self._time_series[f"{strategy}_tokens"].append(
                    MetricPoint(
                        timestamp=timestamp,
                        value=tokens,
                        labels={"strategy": strategy},
                    )
                )

            if cost > 0:
                self._time_series[f"{strategy}_cost"].append(
                    MetricPoint(
                        timestamp=timestamp,
                        value=cost,
                        labels={"strategy": strategy},
                    )
                )

    def get_metrics(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a strategy or all strategies.

        Args:
            strategy: Strategy name, or None for all strategies

        Returns:
            Dictionary of metrics. If strategy is specified, returns metrics
            for that strategy. Otherwise, returns metrics for all strategies.
        """
        with self._lock:
            if strategy:
                if strategy not in self._metrics:
                    return {}
                return self._format_metrics(self._metrics[strategy])
            else:
                return {
                    name: self._format_metrics(metrics)
                    for name, metrics in self._metrics.items()
                }

    def _format_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Format metrics for output.

        Args:
            metrics: Performance metrics to format

        Returns:
            Dictionary of formatted metrics
        """
        return {
            "strategy": metrics.strategy_name,
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
            "failed_queries": metrics.failed_queries,
            "success_rate": round(metrics.success_rate, 2),
            "failure_rate": round(metrics.failure_rate, 2),
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "p50_latency_ms": round(metrics.p50_latency, 2),
            "p95_latency_ms": round(metrics.p95_latency, 2),
            "p99_latency_ms": round(metrics.p99_latency, 2),
            "total_tokens": metrics.total_tokens,
            "total_cost": round(metrics.total_cost, 6),
            "error_count": len(metrics.errors),
            "recent_errors": metrics.errors[-5:]
            if metrics.errors
            else [],  # Last 5 errors
        }

    def get_time_series(
        self, metric_name: str, duration_minutes: int = 60
    ) -> List[MetricPoint]:
        """Get time-series data for a metric.

        Args:
            metric_name: Name of the metric (e.g., "strategy_name_latency")
            duration_minutes: Time window in minutes

        Returns:
            List of metric points within the time window
        """
        with self._lock:
            if metric_name not in self._time_series:
                return []

            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            return [
                point
                for point in self._time_series[metric_name]
                if point.timestamp >= cutoff_time
            ]

    def reset_metrics(self, strategy: Optional[str] = None):
        """Reset metrics for a strategy or all strategies.

        Args:
            strategy: Strategy name to reset, or None to reset all
        """
        with self._lock:
            if strategy:
                if strategy in self._metrics:
                    del self._metrics[strategy]
                # Remove time-series for this strategy
                keys_to_remove = [
                    k for k in self._time_series.keys() if k.startswith(strategy)
                ]
                for key in keys_to_remove:
                    del self._time_series[key]
            else:
                self._metrics.clear()
                self._time_series.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of all metrics.

        Returns:
            Dictionary with aggregated metrics across all strategies
        """
        with self._lock:
            total_queries = sum(
                m.total_queries for m in self._metrics.values()
            )
            total_successful = sum(
                m.successful_queries for m in self._metrics.values()
            )
            total_failed = sum(
                m.failed_queries for m in self._metrics.values()
            )
            total_cost = sum(m.total_cost for m in self._metrics.values())
            total_tokens = sum(m.total_tokens for m in self._metrics.values())
            uptime_seconds = time.time() - self._start_time

            # Calculate overall latency stats
            all_latencies = []
            for metrics in self._metrics.values():
                all_latencies.extend(list(metrics.latencies))

            overall_avg_latency = 0.0
            overall_p50_latency = 0.0
            overall_p95_latency = 0.0
            overall_p99_latency = 0.0

            if all_latencies:
                overall_avg_latency = statistics.mean(all_latencies)
                sorted_latencies = sorted(all_latencies)
                n = len(sorted_latencies)
                overall_p50_latency = sorted_latencies[int(n * 0.5)]
                overall_p95_latency = sorted_latencies[int(n * 0.95)]
                overall_p99_latency = sorted_latencies[int(n * 0.99)]

            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_minutes": round(uptime_seconds / 60, 2),
                "uptime_hours": round(uptime_seconds / 3600, 2),
                "total_queries": total_queries,
                "successful_queries": total_successful,
                "failed_queries": total_failed,
                "overall_success_rate": round(
                    (total_successful / total_queries * 100)
                    if total_queries > 0
                    else 0,
                    2,
                ),
                "total_cost": round(total_cost, 6),
                "total_tokens": total_tokens,
                "queries_per_second": round(
                    total_queries / uptime_seconds if uptime_seconds > 0 else 0,
                    2,
                ),
                "queries_per_minute": round(
                    total_queries / (uptime_seconds / 60)
                    if uptime_seconds > 0
                    else 0,
                    2,
                ),
                "strategies_count": len(self._metrics),
                "avg_latency_ms": round(overall_avg_latency, 2),
                "p50_latency_ms": round(overall_p50_latency, 2),
                "p95_latency_ms": round(overall_p95_latency, 2),
                "p99_latency_ms": round(overall_p99_latency, 2),
            }

    def get_strategy_names(self) -> List[str]:
        """Get list of all strategies being tracked.

        Returns:
            List of strategy names
        """
        with self._lock:
            return list(self._metrics.keys())


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
