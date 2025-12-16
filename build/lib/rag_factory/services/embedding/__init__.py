"""Embedding service for generating text embeddings."""

from .service import EmbeddingService
from .config import EmbeddingServiceConfig
from .base import EmbeddingResult, IEmbeddingProvider

__all__ = [
    "EmbeddingService",
    "EmbeddingServiceConfig",
    "EmbeddingResult",
    "IEmbeddingProvider",
]
