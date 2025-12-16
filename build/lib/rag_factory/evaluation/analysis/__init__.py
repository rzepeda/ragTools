"""
Statistical analysis tools for benchmark results.

This module provides statistical tests and comparison tools for
analyzing and comparing RAG strategy performance.
"""

from rag_factory.evaluation.analysis.statistics import StatisticalAnalyzer
from rag_factory.evaluation.analysis.comparison import StrategyComparator

__all__ = [
    "StatisticalAnalyzer",
    "StrategyComparator",
]
