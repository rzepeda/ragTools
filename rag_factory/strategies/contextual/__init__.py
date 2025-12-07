"""
Contextual Retrieval Strategy.

This package implements a contextual retrieval strategy that enriches
document chunks with LLM-generated contextual descriptions before embedding.
"""

from .config import (
    ContextualRetrievalConfig,
    ContextSource,
    ContextGenerationMethod
)
from .strategy import ContextualRetrievalStrategy
from .context_generator import ContextGenerator
from .batch_processor import BatchProcessor
from .cost_tracker import CostTracker
from .storage import ContextualStorageManager

__all__ = [
    "ContextualRetrievalStrategy",
    "ContextualRetrievalConfig",
    "ContextSource",
    "ContextGenerationMethod",
    "ContextGenerator",
    "BatchProcessor",
    "CostTracker",
    "ContextualStorageManager",
]
