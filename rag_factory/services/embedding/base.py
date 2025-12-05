"""Base embedding provider interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result from embedding generation.

    Attributes:
        embeddings: List of embedding vectors (each is a list of floats)
        model: Name of the model used
        dimensions: Dimension of each embedding vector
        token_count: Total number of tokens processed
        cost: Total cost for generating embeddings
        provider: Name of the provider (openai, cohere, local)
        cached: Boolean flags indicating which embeddings came from cache
    """
    embeddings: List[List[float]]
    model: str
    dimensions: int
    token_count: int
    cost: float
    provider: str
    cached: List[bool]


class IEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different providers.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult containing embeddings and metadata

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions for this model.

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    @abstractmethod
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for this provider.

        Returns:
            Maximum number of texts that can be embedded in one call
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Name of the model being used
        """
        pass

    @abstractmethod
    def calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count.

        Args:
            token_count: Number of tokens processed

        Returns:
            Cost in dollars
        """
        pass
