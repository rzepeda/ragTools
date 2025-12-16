"""Custom exceptions for RAG Factory package."""


class RAGFactoryError(Exception):
    """Base exception for all RAG Factory errors."""

    pass


class StrategyNotFoundError(RAGFactoryError):
    """Raised when a requested strategy is not found in the factory."""

    pass


class ConfigurationError(RAGFactoryError):
    """Raised when there's an error in configuration."""

    pass


class PipelineError(RAGFactoryError):
    """Raised when there's an error in pipeline execution."""

    pass


class InitializationError(RAGFactoryError):
    """Raised when strategy initialization fails."""

    pass


class RetrievalError(RAGFactoryError):
    """Raised when retrieval operation fails."""

    pass


__all__ = [
    "RAGFactoryError",
    "StrategyNotFoundError",
    "ConfigurationError",
    "PipelineError",
    "InitializationError",
    "RetrievalError",
]
