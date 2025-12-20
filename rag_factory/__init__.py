"""RAG Factory package for creating and managing RAG strategies."""

from rag_factory.__version__ import __version__

# Import exceptions - these should be safe
from rag_factory.exceptions import (
    RAGFactoryError,
    StrategyNotFoundError,
    ConfigurationError,
    PipelineError,
    InitializationError,
    RetrievalError,
)

# Import strategies to trigger registration
try:
    import rag_factory.strategies.auto_register
except ImportError:
    pass  # Strategies not available yet

__all__ = [
    "__version__",
    # Exceptions
    "RAGFactoryError",
    "StrategyNotFoundError",
    "ConfigurationError",
    "PipelineError",
    "InitializationError",
    "RetrievalError",
]

# NOTE: Other imports (factory, strategies, pipeline, etc.) are commented out
# to avoid circular import issues. Import them directly when needed:
#   from rag_factory.factory import RAGFactory, register_rag_strategy
#   from rag_factory.strategies.base import IRAGStrategy
# etc.
