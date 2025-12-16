"""Services package for RAG Factory.

This package contains service implementations and interfaces for:
- Embedding services
- LLM services
- Graph database services
- Reranking services
- Database services
- Dependency injection
"""

from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IRerankingService,
    IDatabaseService,
)
from rag_factory.services.dependencies import (
    ServiceDependency,
    StrategyDependencies,
)
from rag_factory.services.consistency import (
    ConsistencyChecker,
)

# NOTE: Service implementations are NOT imported here to avoid circular imports.
# Import them directly when needed:
#   from rag_factory.services.onnx import ONNXEmbeddingService
#   from rag_factory.services.api import OpenAILLMService
#   from rag_factory.services.database import PostgresqlDatabaseService
# etc.

__all__ = [
    # Interfaces
    "ILLMService",
    "IEmbeddingService",
    "IGraphService",
    "IRerankingService",
    "IDatabaseService",
    # Dependency Injection
    "ServiceDependency",
    "StrategyDependencies",
    # Consistency Checking
    "ConsistencyChecker",
]
