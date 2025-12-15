"""RAG Factory package for creating and managing RAG strategies."""

from rag_factory.__version__ import __version__
from rag_factory.exceptions import (
    RAGFactoryError,
    StrategyNotFoundError,
    ConfigurationError,
    PipelineError,
    InitializationError,
    RetrievalError,
)
from rag_factory.factory import (
    RAGFactory,
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
from rag_factory.legacy_config import (
    ConfigManager,
    GlobalConfigSchema,
    StrategyConfigSchema,
    PipelineConfigSchema,
    RAGConfigSchema,
)

from rag_factory.observability import (
    RAGLogger,
    LogContext,
    LogLevel,
    MetricsCollector,
    PerformanceMetrics,
    MetricPoint,
)

__all__ = [
    "__version__",
    # Exceptions
    "RAGFactoryError",
    "StrategyNotFoundError",
    "ConfigurationError",
    "PipelineError",
    "InitializationError",
    "RetrievalError",
    # Factory
    "RAGFactory",
    "register_rag_strategy",
    # Strategies
    "IRAGStrategy",
    "StrategyConfig",
    "Chunk",
    "PreparedData",
    "QueryResult",
    # Pipeline
    "StrategyPipeline",
    "ExecutionMode",
    "PipelineStage",
    "PipelineResult",
    # Config
    "ConfigManager",
    "GlobalConfigSchema",
    "StrategyConfigSchema",
    "PipelineConfigSchema",
    "RAGConfigSchema",
    # Observability
    "RAGLogger",
    "LogContext",
    "LogLevel",
    "MetricsCollector",
    "PerformanceMetrics",
    "MetricPoint",
]


# Auto-register all available strategies
def _register_default_strategies():
    """Auto-register all available built-in strategies."""
    # Register chunking strategies
    try:
        from rag_factory.strategies.chunking import (
            SemanticChunker,
            StructuralChunker,
            HybridChunker,
            FixedSizeChunker,
        )
        RAGFactory.register_strategy("semantic_chunker", SemanticChunker, override=True)
        RAGFactory.register_strategy("structural_chunker", StructuralChunker, override=True)
        RAGFactory.register_strategy("hybrid_chunker", HybridChunker, override=True)
        RAGFactory.register_strategy("fixed_size_chunker", FixedSizeChunker, override=True)
    except ImportError:
        pass  # Chunking strategies not available

    # Register docling chunker if available
    try:
        from rag_factory.strategies.chunking import DoclingChunker, is_docling_available
        if is_docling_available():
            RAGFactory.register_strategy("docling_chunker", DoclingChunker, override=True)
    except ImportError:
        pass  # Docling not available

    # Register reranking strategies
    try:
        from rag_factory.strategies.reranking import (
            CohereReranker,
            CrossEncoderReranker,
            BGEReranker,
        )
        RAGFactory.register_strategy("cohere_reranker", CohereReranker, override=True)
        RAGFactory.register_strategy("cross_encoder_reranker", CrossEncoderReranker, override=True)
        RAGFactory.register_strategy("bge_reranker", BGEReranker, override=True)
    except ImportError:
        pass  # Reranking strategies not available

    # Register query expansion strategies
    try:
        from rag_factory.strategies.query_expansion import (
            HydeExpander,
            LLMExpander,
        )
        RAGFactory.register_strategy("hyde_expander", HydeExpander, override=True)
        RAGFactory.register_strategy("llm_expander", LLMExpander, override=True)
    except ImportError:
        pass  # Query expansion strategies not available


# Auto-register on import
_register_default_strategies()
