"""Chunking strategies for document processing.

This module provides various strategies for splitting documents into chunks:

- SemanticChunker: Chunks based on semantic similarity using embeddings
- StructuralChunker: Chunks based on document structure (headers, paragraphs)
- HybridChunker: Combines structural and semantic approaches
- FixedSizeChunker: Simple fixed-size chunking (baseline)

Example usage:
    ```python
    from rag_factory.strategies.chunking import (
        StructuralChunker,
        ChunkingConfig,
        ChunkingMethod
    )

    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512
    )

    chunker = StructuralChunker(config)
    chunks = chunker.chunk_document(document_text, "doc_id")
    ```
"""

from .base import (
    IChunker,
    Chunk,
    ChunkMetadata,
    ChunkingConfig,
    ChunkingMethod
)

from .semantic_chunker import SemanticChunker
from .structural_chunker import StructuralChunker
from .hybrid_chunker import HybridChunker
from .fixed_size_chunker import FixedSizeChunker

# Optional: Import docling chunker if available
try:
    from .docling_chunker import DoclingChunker, is_docling_available
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    DoclingChunker = None
    is_docling_available = lambda: False

__all__ = [
    # Base classes
    "IChunker",
    "Chunk",
    "ChunkMetadata",
    "ChunkingConfig",
    "ChunkingMethod",

    # Chunking strategies
    "SemanticChunker",
    "StructuralChunker",
    "HybridChunker",
    "FixedSizeChunker",
]

# Add docling chunker to exports if available
if DOCLING_AVAILABLE:
    __all__.extend(["DoclingChunker", "is_docling_available"])
