"""
Data models for late chunking strategy.

This module defines all data structures used in the late chunking RAG strategy,
including token embeddings, document embeddings, chunks, and configuration.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingChunkingMethod(str, Enum):
    """Methods for chunking embeddings."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class TokenEmbedding(BaseModel):
    """Token-level embedding with text mapping."""
    token: str
    token_id: int
    start_char: int
    end_char: int
    embedding: List[float]
    position: int

    class Config:
        arbitrary_types_allowed = True


class DocumentEmbedding(BaseModel):
    """Full document embedding with token details."""
    document_id: str
    text: str
    full_embedding: List[float]  # Document-level embedding
    token_embeddings: List[TokenEmbedding]  # Token-level embeddings
    model_name: str
    token_count: int
    embedding_dim: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class EmbeddingChunk(BaseModel):
    """Chunk created from embedding split."""
    chunk_id: str
    document_id: str
    text: str
    chunk_embedding: List[float]  # Average of token embeddings in chunk
    token_range: Tuple[int, int]  # Start and end token indices
    char_range: Tuple[int, int]  # Start and end character positions
    token_count: int
    coherence_score: Optional[float] = None
    chunking_method: EmbeddingChunkingMethod
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class LateChunkingConfig(BaseModel):
    """Configuration for late chunking."""
    # Document embedding
    use_long_context_model: bool = True
    max_document_tokens: int = 16384
    model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # Chunking
    chunking_method: EmbeddingChunkingMethod = EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
    target_chunk_size: int = 512
    min_chunk_size: int = 128
    max_chunk_size: int = 1024
    chunk_overlap_tokens: int = 50

    # Semantic boundary detection
    similarity_threshold: float = 0.7
    use_local_similarity: bool = True

    # Coherence analysis
    compute_coherence_scores: bool = True
    coherence_window_size: int = 3

    # Performance
    batch_size: int = 1
    device: str = "cpu"  # or "cuda"


class CoherenceMetrics(BaseModel):
    """Coherence metrics for chunk evaluation."""
    intra_chunk_similarity: float  # Average similarity within chunk
    inter_chunk_similarity: float  # Similarity with adjacent chunks
    variance: float  # Embedding variance within chunk
    semantic_boundary_score: float  # How well boundaries align with semantics
    comparison_to_traditional: Optional[float] = None  # Improvement over traditional
