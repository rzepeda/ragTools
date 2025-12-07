"""
Late chunking strategy package.

This package implements the late chunking RAG strategy which embeds
full documents before chunking to maintain better context.
"""

from .strategy import LateChunkingRAGStrategy
from .models import (
    EmbeddingChunkingMethod,
    TokenEmbedding,
    DocumentEmbedding,
    EmbeddingChunk,
    LateChunkingConfig,
    CoherenceMetrics
)
from .document_embedder import DocumentEmbedder
from .embedding_chunker import EmbeddingChunker
from .coherence_analyzer import CoherenceAnalyzer

__all__ = [
    "LateChunkingRAGStrategy",
    "EmbeddingChunkingMethod",
    "TokenEmbedding",
    "DocumentEmbedding",
    "EmbeddingChunk",
    "LateChunkingConfig",
    "CoherenceMetrics",
    "DocumentEmbedder",
    "EmbeddingChunker",
    "CoherenceAnalyzer",
]
