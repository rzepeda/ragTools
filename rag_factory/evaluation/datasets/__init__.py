"""
Dataset management for evaluation.

This module provides tools for loading, managing, and analyzing
evaluation datasets.
"""

from rag_factory.evaluation.datasets.schema import (
    EvaluationExample,
    EvaluationDataset,
)
from rag_factory.evaluation.datasets.loader import DatasetLoader

__all__ = [
    "EvaluationExample",
    "EvaluationDataset",
    "DatasetLoader",
]
