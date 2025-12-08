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

# ONNX Services
from rag_factory.services.onnx import (
    ONNXEmbeddingService,
)

# API Services
from rag_factory.services.api import (
    AnthropicLLMService,
    OpenAILLMService,
    OpenAIEmbeddingService,
    CohereRerankingService,
)

# Database Services
from rag_factory.services.database import (
    Neo4jGraphService,
    PostgresqlDatabaseService,
)

# Local Services
from rag_factory.services.local import (
    CosineRerankingService,
)

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
    # ONNX Services
    "ONNXEmbeddingService",
    # API Services
    "AnthropicLLMService",
    "OpenAILLMService",
    "OpenAIEmbeddingService",
    "CohereRerankingService",
    # Database Services
    "Neo4jGraphService",
    "PostgresqlDatabaseService",
    # Local Services
    "CosineRerankingService",
]


