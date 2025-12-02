"""RAG Factory package for creating and managing RAG strategies."""

from rag_factory.factory import (
    RAGFactory,
    StrategyNotFoundError,
    ConfigurationError,
    register_rag_strategy,
)
from rag_factory.strategies.base import (
    IRAGStrategy,
    StrategyConfig,
    Chunk,
    PreparedData,
    QueryResult,
)
from rag_factory.pipeline import (
    StrategyPipeline,
    ExecutionMode,
    PipelineStage,
    PipelineResult,
)

__all__ = [
    "RAGFactory",
    "StrategyNotFoundError",
    "ConfigurationError",
    "register_rag_strategy",
    "IRAGStrategy",
    "StrategyConfig",
    "Chunk",
    "PreparedData",
    "QueryResult",
    "StrategyPipeline",
    "ExecutionMode",
    "PipelineStage",
    "PipelineResult",
]
