"""
RAG Factory Evaluation Framework.

This module provides a comprehensive evaluation framework for comparing
and analyzing RAG (Retrieval-Augmented Generation) strategies.

Features:
- Multiple evaluation metrics (retrieval, quality, performance, cost)
- Dataset management and loading
- Benchmark runner for strategy comparison
- Statistical analysis and significance testing
- Results visualization and export

Example:
    >>> from rag_factory.evaluation import BenchmarkRunner, PrecisionAtK, RecallAtK
    >>> from rag_factory.evaluation.datasets import DatasetLoader
    >>>
    >>> # Load dataset
    >>> loader = DatasetLoader()
    >>> dataset = loader.load("path/to/dataset.json")
    >>>
    >>> # Run benchmark
    >>> metrics = [PrecisionAtK(k=5), RecallAtK(k=5)]
    >>> runner = BenchmarkRunner(metrics)
    >>> results = runner.run(strategy, dataset)
    >>> print(results.aggregate_metrics)
"""

__version__ = "0.1.0"

from rag_factory.evaluation.metrics.base import (
    IMetric,
    MetricResult,
    MetricType,
)
from rag_factory.evaluation.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkResult,
)

__all__ = [
    "IMetric",
    "MetricResult",
    "MetricType",
    "BenchmarkRunner",
    "BenchmarkResult",
]
