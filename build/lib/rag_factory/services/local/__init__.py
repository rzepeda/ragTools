"""Local service implementations.

This package provides service implementations that run locally
without external API calls.
"""

from rag_factory.services.local.reranker import CosineRerankingService

__all__ = [
    "CosineRerankingService",
]
