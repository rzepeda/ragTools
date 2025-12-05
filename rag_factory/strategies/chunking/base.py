"""Base classes and interfaces for document chunking strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ChunkingMethod(Enum):
    """Enumeration of chunking methods."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    FIXED_SIZE = "fixed_size"
    DOCKLING = "dockling"


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk
        source_document_id: ID of the source document
        position: Position in document (0-indexed)
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        section_hierarchy: List of section headers (e.g., ["Chapter 1", "Section 1.1"])
        chunking_method: Method used to create this chunk
        token_count: Number of tokens in the chunk
        coherence_score: Optional semantic coherence score (0.0-1.0)
        parent_chunk_id: Optional parent chunk ID for hierarchical chunks
        metadata: Additional custom metadata
    """
    chunk_id: str
    source_document_id: str
    position: int
    start_char: int
    end_char: int
    section_hierarchy: List[str]
    chunking_method: ChunkingMethod
    token_count: int
    coherence_score: Optional[float] = None
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A document chunk with text and metadata.

    Attributes:
        text: The chunk text content
        metadata: Chunk metadata
        embedding: Optional embedding vector for the chunk
    """
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies.

    Attributes:
        method: Chunking method to use
        min_chunk_size: Minimum chunk size in tokens
        max_chunk_size: Maximum chunk size in tokens
        target_chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

        similarity_threshold: Threshold for semantic boundary detection (0.0-1.0)
        use_embeddings: Whether to use embeddings for semantic chunking

        respect_headers: Whether to respect markdown headers
        respect_paragraphs: Whether to respect paragraph boundaries
        keep_code_blocks_intact: Whether to keep code blocks as atomic units
        keep_tables_intact: Whether to keep tables as atomic units

        use_dockling: Whether to use dockling for advanced parsing
        dockling_fallback: Whether to fallback to basic chunking if dockling fails

        compute_coherence_scores: Whether to compute coherence scores
        preserve_metadata: Whether to preserve metadata
        extra_config: Additional custom configuration
    """
    method: ChunkingMethod = ChunkingMethod.HYBRID
    min_chunk_size: int = 128
    max_chunk_size: int = 1024
    target_chunk_size: int = 512
    chunk_overlap: int = 50

    # Semantic chunking settings
    similarity_threshold: float = 0.7
    use_embeddings: bool = True

    # Structural chunking settings
    respect_headers: bool = True
    respect_paragraphs: bool = True
    keep_code_blocks_intact: bool = True
    keep_tables_intact: bool = True

    # Dockling settings
    use_dockling: bool = False  # Disabled by default since dockling not available
    dockling_fallback: bool = True

    # General settings
    compute_coherence_scores: bool = True
    preserve_metadata: bool = True
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) cannot be greater than "
                f"max_chunk_size ({self.max_chunk_size})"
            )

        if self.target_chunk_size < self.min_chunk_size or self.target_chunk_size > self.max_chunk_size:
            raise ValueError(
                f"target_chunk_size ({self.target_chunk_size}) must be between "
                f"min_chunk_size ({self.min_chunk_size}) and max_chunk_size ({self.max_chunk_size})"
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold ({self.similarity_threshold}) must be between 0.0 and 1.0"
            )


class IChunker(ABC):
    """Abstract base class for document chunking strategies.

    All chunking strategies must inherit from this class and implement
    the required methods.
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration
        """
        self.config = config

    @abstractmethod
    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk a document into semantically coherent pieces.

        Args:
            document: The document text to chunk
            document_id: Unique identifier for the document

        Returns:
            List of Chunk objects with text and metadata
        """
        pass

    @abstractmethod
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents in batch.

        Args:
            documents: List of dicts with 'text' and 'id' keys

        Returns:
            List of chunk lists, one per document
        """
        pass

    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """Validate that chunks meet quality criteria.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if all chunks are valid, False otherwise
        """
        for chunk in chunks:
            token_count = chunk.metadata.token_count

            # Check size constraints
            if token_count < self.config.min_chunk_size:
                # Allow small chunks for atomic content
                if not self._is_atomic_content(chunk.text):
                    return False

            if token_count > self.config.max_chunk_size:
                # Allow oversized chunks for atomic content
                if not self._is_atomic_content(chunk.text):
                    return False

        return True

    def _is_atomic_content(self, text: str) -> bool:
        """Check if content should be kept as atomic unit (code, table, etc.).

        Args:
            text: Text to check

        Returns:
            True if content is atomic, False otherwise
        """
        text_stripped = text.strip()

        # Check for code blocks
        if text_stripped.startswith("```") or text_stripped.startswith("    "):
            return True

        # Check for tables (simple heuristic)
        if "|" in text and text.count("|") > 3:
            return True

        return False

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunk distribution.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "chunking_method": self.config.method.value
            }

        sizes = [c.metadata.token_count for c in chunks]
        coherence_scores = [
            c.metadata.coherence_score
            for c in chunks
            if c.metadata.coherence_score is not None
        ]

        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "chunking_method": self.config.method.value
        }

        if coherence_scores:
            stats["avg_coherence"] = sum(coherence_scores) / len(coherence_scores)
            stats["min_coherence"] = min(coherence_scores)
            stats["max_coherence"] = max(coherence_scores)
        else:
            stats["avg_coherence"] = None

        return stats
