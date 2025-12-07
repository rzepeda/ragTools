"""
Hierarchical RAG Strategy Models.

This module defines data structures for hierarchical chunk relationships,
including hierarchy levels, metadata, and expansion strategies.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import List, Dict, Any, Optional
from uuid import UUID


class HierarchyLevel(IntEnum):
    """Hierarchy levels for document structure.
    
    Lower values represent higher levels in the hierarchy.
    """
    DOCUMENT = 0
    SECTION = 1
    PARAGRAPH = 2
    SENTENCE = 3


class ExpansionStrategy(str, Enum):
    """Strategies for expanding chunks with parent context."""
    
    IMMEDIATE_PARENT = "immediate_parent"
    """Include only the direct parent chunk."""
    
    FULL_SECTION = "full_section"
    """Include all ancestors up to section level."""
    
    WINDOW = "window"
    """Include N siblings before and after the chunk."""
    
    FULL_DOCUMENT = "full_document"
    """Include the entire document as context."""
    
    ADAPTIVE = "adaptive"
    """Adaptively choose strategy based on chunk size and relevance score."""


@dataclass
class HierarchyMetadata:
    """Metadata about a chunk's position in the hierarchy.
    
    Attributes:
        position_in_parent: 0-indexed position among siblings
        total_siblings: Total number of sibling chunks
        depth_from_root: Distance from root chunk (0 for root)
    """
    position_in_parent: int
    total_siblings: int
    depth_from_root: int


@dataclass
class HierarchicalChunk:
    """A text chunk with hierarchical relationships.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Identifier for the parent document
        text: The text content of the chunk
        hierarchy_level: Level in the hierarchy (0=document, 1=section, etc.)
        hierarchy_metadata: Metadata about position in hierarchy
        parent_chunk_id: ID of parent chunk (None for root)
        token_count: Number of tokens in the chunk
        metadata: Additional metadata
        embedding: Vector embedding (optional)
    """
    chunk_id: str
    document_id: str
    text: str
    hierarchy_level: HierarchyLevel
    hierarchy_metadata: HierarchyMetadata
    parent_chunk_id: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ChunkHierarchy:
    """Represents the complete hierarchy of chunks for a document.
    
    Attributes:
        document_id: Identifier for the document
        root_chunk: The root chunk of the hierarchy
        all_chunks: Dictionary mapping chunk_id to HierarchicalChunk
        levels: Dictionary mapping hierarchy level to list of chunk IDs
    """
    document_id: str
    root_chunk: HierarchicalChunk
    all_chunks: Dict[str, HierarchicalChunk]
    levels: Dict[HierarchyLevel, List[str]]


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical RAG strategy.
    
    Attributes:
        expansion_strategy: Strategy for expanding chunks with parent context
        small_chunk_size: Size of small chunks for precise search (in tokens)
        large_chunk_size: Size of large parent chunks for context (in tokens)
        search_small_chunks: Whether to search on small chunks (vs large)
        window_size: Number of siblings to include for WINDOW strategy
        max_hierarchy_depth: Maximum depth of hierarchy to build
        min_chunk_size: Minimum chunk size to avoid tiny chunks
    """
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.IMMEDIATE_PARENT
    small_chunk_size: int = 256
    large_chunk_size: int = 1024
    search_small_chunks: bool = True
    window_size: int = 2
    max_hierarchy_depth: int = 4
    min_chunk_size: int = 50


@dataclass
class ExpandedChunk:
    """A chunk that has been expanded with parent context.
    
    Attributes:
        original_chunk: The original small chunk that was retrieved
        expanded_text: The text after expansion with parent context
        expansion_strategy: The strategy used for expansion
        parent_chunks: List of parent chunks included in expansion
        total_tokens: Total token count after expansion
    """
    original_chunk: HierarchicalChunk
    expanded_text: str
    expansion_strategy: ExpansionStrategy
    parent_chunks: List[HierarchicalChunk]
    total_tokens: int
