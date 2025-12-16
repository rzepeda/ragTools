"""
Base metric interface for evaluation framework.

This module defines the abstract base class and data structures that all
evaluation metrics must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of evaluation metrics."""
    RETRIEVAL = "retrieval"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COST = "cost"


@dataclass
class MetricResult:
    """
    Result from a metric computation.

    Attributes:
        name: Metric name
        value: Computed metric value
        metadata: Additional information about the computation
        query_id: Optional identifier for the query
    """
    name: str
    value: float
    metadata: Dict[str, Any]
    query_id: Optional[str] = None


class IMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    All metrics should inherit from this class and implement
    the compute method.

    Attributes:
        name: Human-readable metric name
        metric_type: Type of metric (retrieval, quality, performance, cost)

    Example:
        >>> class MyMetric(IMetric):
        ...     def __init__(self):
        ...         super().__init__("my_metric", MetricType.RETRIEVAL)
        ...
        ...     def compute(self, **kwargs) -> MetricResult:
        ...         # Computation logic here
        ...         return MetricResult(
        ...             name=self.name,
        ...             value=0.85,
        ...             metadata={"details": "..."}
        ...         )
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Description of my metric"
    """

    def __init__(self, name: str, metric_type: MetricType):
        """
        Initialize metric.

        Args:
            name: Metric name
            metric_type: Type of metric
        """
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """
        Compute the metric value.

        Args:
            **kwargs: Metric-specific inputs

        Returns:
            MetricResult with computed value and metadata

        Raises:
            ValueError: If required inputs are missing or invalid
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the metric."""
        pass

    @property
    def higher_is_better(self) -> bool:
        """
        Whether higher values are better (default: True).

        Override in subclass if lower values are better.

        Returns:
            True if higher values are better, False otherwise
        """
        return True

    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}', type={self.metric_type.value})"
