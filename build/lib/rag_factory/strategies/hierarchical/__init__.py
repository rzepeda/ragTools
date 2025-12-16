"""
Hierarchical RAG Strategy.

This package implements hierarchical chunk relationships for RAG,
enabling parent-child context expansion during retrieval.
"""

from .models import (
    HierarchyLevel,
    ExpansionStrategy,
    HierarchyMetadata,
    HierarchicalChunk,
    ChunkHierarchy,
    HierarchicalConfig,
    ExpandedChunk
)
from .hierarchy_builder import HierarchyBuilder
from .parent_retriever import ParentRetriever
from .strategy import HierarchicalRAGStrategy

__all__ = [
    "HierarchyLevel",
    "ExpansionStrategy",
    "HierarchyMetadata",
    "HierarchicalChunk",
    "ChunkHierarchy",
    "HierarchicalConfig",
    "ExpandedChunk",
    "HierarchyBuilder",
    "ParentRetriever",
    "HierarchicalRAGStrategy",
]
