"""OpenAI embedding provider implementation."""

from typing import List, Dict, Any
import logging

try:
    import openai
    from tenacity import retry, stop_after_attempt, wait_exponential
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
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


class OpenAIProvider(IEmbeddingProvider):
    """OpenAI embedding provider.

    Supports OpenAI's embedding models including:
    - text-embedding-3-small
    - text-embedding-3-large
    - text-embedding-ada-002
    """

    MODELS = {
        "text-embedding-3-small": {"dimensions": 1536, "cost_per_1k": 0.00002},
        "text-embedding-3-large": {"dimensions": 3072, "cost_per_1k": 0.00013},
        "text-embedding-ada-002": {"dimensions": 1536, "cost_per_1k": 0.0001}
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider.

        Args:
            config: Configuration dictionary with 'api_key', 'model', and
                   optional 'max_batch_size'

        Raises:
            ImportError: If openai package is not installed
            ValueError: If model is not supported
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai tenacity"
            )

        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-embedding-3-small")
        self.max_batch_size = config.get("max_batch_size", 100)

        if self.model not in self.MODELS:
            raise ValueError(
                f"Unknown OpenAI model: {self.model}. "
                f"Supported models: {list(self.MODELS.keys())}"
            )

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            Exception: If API call fails after retries
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            token_count = response.usage.total_tokens
            cost = self.calculate_cost(token_count)

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.get_dimensions(),
                token_count=token_count,
                cost=cost,
                provider="openai",
                cached=[False] * len(texts)
            )
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
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
