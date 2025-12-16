"""Multi-Query RAG Strategy.

This strategy generates multiple query variants using an LLM, executes them
in parallel, deduplicates results, and ranks them using various strategies
(RRF, frequency boost, max score, hybrid) to improve retrieval coverage.
"""

from .config import (
    VariantType,
    RankingStrategy,
    MultiQueryConfig,
)
from .variant_generator import QueryVariantGenerator
from .parallel_executor import ParallelQueryExecutor
from .deduplicator import ResultDeduplicator
from .ranker import ResultRanker
from .strategy import MultiQueryRAGStrategy

__all__ = [
    "VariantType",
    "RankingStrategy",
    "MultiQueryConfig",
    "QueryVariantGenerator",
    "ParallelQueryExecutor",
    "ResultDeduplicator",
    "ResultRanker",
    "MultiQueryRAGStrategy",
]

