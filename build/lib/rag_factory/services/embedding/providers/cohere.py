"""Cohere embedding provider implementation."""

from typing import List, Dict, Any
import logging

try:
    import cohere
    from tenacity import retry, stop_after_attempt, wait_exponential
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    # Provide a no-op decorator when tenacity is not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def stop_after_attempt(*args, **kwargs):
        pass
    def wait_exponential(*args, **kwargs):
        pass

from ..base import IEmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)


class CohereProvider(IEmbeddingProvider):
    """Cohere embedding provider.

    Supports Cohere's embedding models including:
    - embed-english-v3.0
    - embed-multilingual-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-light-v3.0
    """

    MODELS = {
        "embed-english-v3.0": {"dimensions": 1024, "cost_per_1k": 0.0001},
        "embed-multilingual-v3.0": {"dimensions": 1024, "cost_per_1k": 0.0001},
        "embed-english-light-v3.0": {"dimensions": 384, "cost_per_1k": 0.0001},
        "embed-multilingual-light-v3.0": {"dimensions": 384, "cost_per_1k": 0.0001},
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize Cohere provider.

        Args:
            config: Configuration dictionary with 'api_key', 'model', and
                   optional 'max_batch_size', 'input_type'

        Raises:
            ImportError: If cohere package is not installed
            ValueError: If model is not supported
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere package not installed. "
                "Install with: pip install cohere tenacity"
            )

        self.api_key = config.get("api_key")
        self.model = config.get("model", "embed-english-v3.0")
        self.max_batch_size = config.get("max_batch_size", 96)
        self.input_type = config.get("input_type", "search_document")

        if self.model not in self.MODELS:
            raise ValueError(
                f"Unknown Cohere model: {self.model}. "
                f"Supported models: {list(self.MODELS.keys())}"
            )

        if not self.api_key:
            raise ValueError("Cohere API key is required")

        self.client = cohere.Client(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using Cohere API.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            Exception: If API call fails after retries
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type
            )

            embeddings = response.embeddings

            # Cohere doesn't return token count, estimate it
            # Rough estimate: ~0.75 tokens per word
            token_count = sum(len(text.split()) for text in texts)
            token_count = int(token_count * 0.75)

            cost = self.calculate_cost(token_count)

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.get_dimensions(),
                token_count=token_count,
                cost=cost,
                provider="cohere",
                cached=[False] * len(texts)
            )
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions for the current model.

        Returns:
            Number of dimensions in embedding vectors
        """
        return self.MODELS[self.model]["dimensions"]

    def get_max_batch_size(self) -> int:
        """Get maximum batch size.

        Returns:
            Maximum number of texts that can be embedded in one call
        """
        return self.max_batch_size

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Name of the model being used
        """
        return self.model

    def calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count.

        Args:
            token_count: Number of tokens processed

        Returns:
            Cost in dollars
        """
        cost_per_1k = self.MODELS[self.model]["cost_per_1k"]
        return (token_count / 1000.0) * cost_per_1k
