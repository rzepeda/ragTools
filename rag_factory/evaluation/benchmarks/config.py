"""
Configuration for benchmark runs.

This module provides configuration classes for benchmark execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from rag_factory.evaluation.metrics.base import IMetric


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run.

    Attributes:
        metrics: List of metrics to compute
        top_k: Number of results to retrieve (default: 10)
        batch_size: Batch size for parallel processing (default: 1)
        enable_caching: Whether to cache results (default: True)
        cache_dir: Directory for caching results (default: ".benchmark_cache")
        checkpoint_interval: Save checkpoint every N queries (default: 50)
        enable_checkpointing: Whether to enable checkpointing (default: False)
        verbose: Print progress information (default: True)
        metadata: Additional configuration metadata

    Example:
        >>> from rag_factory.evaluation.metrics import PrecisionAtK, RecallAtK
        >>> config = BenchmarkConfig(
        ...     metrics=[PrecisionAtK(k=5), RecallAtK(k=5)],
        ...     top_k=10,
        ...     verbose=True
        ... )
    """
    metrics: List[IMetric]
    top_k: int = 10
    batch_size: int = 1
    enable_caching: bool = True
    cache_dir: str = ".benchmark_cache"
    checkpoint_interval: int = 50
    enable_checkpointing: bool = False
    verbose: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.metrics:
            raise ValueError("At least one metric must be provided")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {self.checkpoint_interval}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "metric_names": [m.name for m in self.metrics],
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching,
            "cache_dir": self.cache_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "enable_checkpointing": self.enable_checkpointing,
            "verbose": self.verbose,
            "metadata": self.metadata
        }
