"""
Performance evaluation metrics.

This module provides metrics for evaluating system performance,
including latency and throughput measurements.
"""

from typing import Optional, Dict, Any
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType


class Latency(IMetric):
    """
    Latency: Time taken to complete an operation.

    Measures the response time in milliseconds. Can be used for
    different operation types (retrieval, generation, end-to-end).

    Args:
        operation_type: Type of operation being measured (default: "total")

    Example:
        >>> metric = Latency(operation_type="retrieval")
        >>> result = metric.compute(latency_ms=150.5)
        >>> print(f"Latency: {result.value}ms")
    """

    def __init__(self, operation_type: str = "total"):
        """
        Initialize Latency metric.

        Args:
            operation_type: Type of operation (e.g., "retrieval", "generation", "total")
        """
        super().__init__(f"{operation_type}_latency_ms", MetricType.PERFORMANCE)
        self.operation_type = operation_type

    def compute(
        self,
        latency_ms: float,
        query_id: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute latency metric.

        Args:
            latency_ms: Latency in milliseconds
            query_id: Optional query identifier
            **kwargs: Additional metadata

        Returns:
            MetricResult with latency value
        """
        return MetricResult(
            name=self.name,
            value=latency_ms,
            metadata={
                "operation_type": self.operation_type,
                "unit": "milliseconds",
                **kwargs
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Time taken for {self.operation_type} operation in milliseconds"

    @property
    def higher_is_better(self) -> bool:
        """Lower latency is better."""
        return False


class Throughput(IMetric):
    """
    Throughput: Number of operations per second.

    Measures the rate at which operations can be processed.

    Example:
        >>> metric = Throughput()
        >>> result = metric.compute(queries_per_second=12.5)
        >>> print(f"Throughput: {result.value} queries/sec")
    """

    def __init__(self):
        """Initialize Throughput metric."""
        super().__init__("throughput_qps", MetricType.PERFORMANCE)

    def compute(
        self,
        queries_per_second: float,
        total_queries: Optional[int] = None,
        total_time_seconds: Optional[float] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute throughput metric.

        Args:
            queries_per_second: Queries processed per second
            total_queries: Total number of queries processed
            total_time_seconds: Total time taken
            **kwargs: Additional metadata

        Returns:
            MetricResult with throughput value
        """
        metadata = {
            "unit": "queries_per_second",
            **kwargs
        }
        if total_queries is not None:
            metadata["total_queries"] = total_queries
        if total_time_seconds is not None:
            metadata["total_time_seconds"] = total_time_seconds

        return MetricResult(
            name=self.name,
            value=queries_per_second,
            metadata=metadata,
            query_id=None  # Throughput is typically aggregate
        )

    @property
    def description(self) -> str:
        return "Number of queries processed per second"


class PercentileLatency(IMetric):
    """
    Percentile Latency: Latency at a specific percentile.

    Measures latency at a given percentile (e.g., P50, P95, P99).
    Useful for understanding tail latencies.

    Args:
        percentile: Percentile to compute (0-100, default: 95)
        operation_type: Type of operation (default: "total")

    Example:
        >>> metric = PercentileLatency(percentile=95)
        >>> result = metric.compute(latency_ms=250.0)
        >>> print(f"P95 Latency: {result.value}ms")
    """

    def __init__(self, percentile: int = 95, operation_type: str = "total"):
        """
        Initialize Percentile Latency metric.

        Args:
            percentile: Percentile to compute (0-100)
            operation_type: Type of operation

        Raises:
            ValueError: If percentile is not in [0, 100]
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        super().__init__(f"p{percentile}_{operation_type}_latency_ms", MetricType.PERFORMANCE)
        self.percentile = percentile
        self.operation_type = operation_type

    def compute(
        self,
        latency_ms: float,
        query_id: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute percentile latency.

        Args:
            latency_ms: Latency value at the specified percentile
            query_id: Optional query identifier
            **kwargs: Additional metadata

        Returns:
            MetricResult with percentile latency value
        """
        return MetricResult(
            name=self.name,
            value=latency_ms,
            metadata={
                "percentile": self.percentile,
                "operation_type": self.operation_type,
                "unit": "milliseconds",
                **kwargs
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"P{self.percentile} latency for {self.operation_type} operation"

    @property
    def higher_is_better(self) -> bool:
        """Lower latency is better."""
        return False
