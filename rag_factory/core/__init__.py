"""Core models and enums for RAG Factory.

This module provides core data structures and enumerations used throughout
the RAG Factory system, including capability definitions and result models.
"""

from rag_factory.core.capabilities import (
    IndexCapability,
    IndexingResult,
    ValidationResult,
)
from rag_factory.core.indexing_interface import (
    IndexingContext,
    IIndexingStrategy,
    VectorEmbeddingIndexing,
)
from rag_factory.core.retrieval_interface import (
    RetrievalContext,
    IRetrievalStrategy,
    RerankingRetrieval,
)

__all__ = [
    "IndexCapability",
    "IndexingResult",
    "ValidationResult",
    "IndexingContext",
    "IIndexingStrategy",
    "VectorEmbeddingIndexing",
    "RetrievalContext",
    "IRetrievalStrategy",
    "RerankingRetrieval",
]
