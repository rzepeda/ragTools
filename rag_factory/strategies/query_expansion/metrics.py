"""Metrics tracking for query expansion quality."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class ExpansionMetrics:
    """Metrics for tracking expansion quality."""
    query_id: str
    original_query: str
    expanded_query: str
    strategy: str
    execution_time_ms: float
    cache_hit: bool
    num_added_terms: int
    llm_tokens_used: int = 0
    llm_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple expansions."""
    total_expansions: int
    avg_execution_time_ms: float
    median_execution_time_ms: float
    p95_execution_time_ms: float
    cache_hit_rate: float
    avg_added_terms: float
    total_llm_tokens: int
    total_llm_cost: float
    strategies_used: Dict[str, int]


class MetricsTracker:
    """Tracks and aggregates query expansion metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self._metrics: List[ExpansionMetrics] = []
        self._max_stored = 10000  # Limit memory usage

    def record(self, metric: ExpansionMetrics) -> None:
        """Record a single expansion metric.

        Args:
            metric: Expansion metrics to record
        """
        self._metrics.append(metric)

        # Trim old metrics if exceeding limit
        if len(self._metrics) > self._max_stored:
            self._metrics = self._metrics[-self._max_stored:]

    def get_aggregated_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None
    ) -> AggregatedMetrics:
        """Get aggregated metrics for a time period.

        Args:
            start_time: Start of time window
            end_time: End of time window
            strategy: Filter by specific strategy

        Returns:
            AggregatedMetrics with aggregated statistics
        """
        # Filter metrics
        filtered = self._metrics

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        if strategy:
            filtered = [m for m in filtered if m.strategy == strategy]

        if not filtered:
            return AggregatedMetrics(
                total_expansions=0,
                avg_execution_time_ms=0.0,
                median_execution_time_ms=0.0,
                p95_execution_time_ms=0.0,
                cache_hit_rate=0.0,
                avg_added_terms=0.0,
                total_llm_tokens=0,
                total_llm_cost=0.0,
                strategies_used={}
            )

        # Calculate aggregated metrics
        execution_times = [m.execution_time_ms for m in filtered]
        cache_hits = sum(1 for m in filtered if m.cache_hit)
        added_terms = [m.num_added_terms for m in filtered]

        strategies_used: Dict[str, int] = {}
        for m in filtered:
            strategies_used[m.strategy] = strategies_used.get(m.strategy, 0) + 1

        return AggregatedMetrics(
            total_expansions=len(filtered),
            avg_execution_time_ms=statistics.mean(execution_times),
            median_execution_time_ms=statistics.median(execution_times),
            p95_execution_time_ms=self._percentile(execution_times, 95),
            cache_hit_rate=cache_hits / len(filtered),
            avg_added_terms=statistics.mean(added_terms) if added_terms else 0.0,
            total_llm_tokens=sum(m.llm_tokens_used for m in filtered),
            total_llm_cost=sum(m.llm_cost for m in filtered),
            strategies_used=strategies_used
        )

    def get_recent_metrics(self, count: int = 100) -> List[ExpansionMetrics]:
        """Get most recent expansion metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent metrics
        """
        return self._metrics[-count:]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._metrics.clear()

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Value at the given percentile
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_stats(self) -> Dict[str, Any]:
        """Get current tracker statistics.

        Returns:
            Dictionary with tracker statistics
        """
        return {
            "total_metrics_stored": len(self._metrics),
            "max_stored": self._max_stored,
            "oldest_metric": self._metrics[0].timestamp if self._metrics else None,
            "newest_metric": self._metrics[-1].timestamp if self._metrics else None
        }
