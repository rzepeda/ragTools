"""Main embedding service implementation."""

from typing import List, Optional, Dict, Any
import hashlib
import logging

from .base import IEmbeddingProvider, EmbeddingResult
from .cache import EmbeddingCache
from .rate_limiter import RateLimiter
from .config import EmbeddingServiceConfig
from .providers import OpenAIProvider, CohereProvider, LocalProvider, ONNXLocalProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Centralized embedding service with caching and rate limiting.

    This service provides a unified interface for generating embeddings
    from multiple providers (OpenAI, Cohere, local models) with built-in
    caching, rate limiting, and batch processing.

    Example:
        ```python
        config = EmbeddingServiceConfig(
            provider="openai",
            model="text-embedding-3-small"
        )
        service = EmbeddingService(config)
        result = service.embed(["Hello world", "Another text"])
        embeddings = result.embeddings
        ```

    Attributes:
        config: Service configuration
        provider: Embedding provider instance
        cache: Cache instance (if enabled)
        rate_limiter: Rate limiter instance (if enabled)
    """

    def __init__(self, config: EmbeddingServiceConfig):
        """Initialize the embedding service.

        Args:
            config: Service configuration

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        self.config = config
        self.provider = self._init_provider()
        self.cache = (
            EmbeddingCache(config.cache_config)
            if config.enable_cache
            else None
        )
        self.rate_limiter = (
            RateLimiter(config.rate_limit_config)
            if config.enable_rate_limiting
            else None
        )
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }

    def _init_provider(self) -> IEmbeddingProvider:
        """Initialize the embedding provider based on config.

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider is unknown
        """
        provider_map = {
            "openai": OpenAIProvider,
            "cohere": CohereProvider,
            "local": LocalProvider,
            "onnx-local": ONNXLocalProvider
        }

        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {self.config.provider}. "
                f"Supported providers: {list(provider_map.keys())}"
            )

        return provider_class(self.config.provider_config)

    def embed(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> EmbeddingResult:
        """Generate embeddings for list of texts.

        This method handles caching, rate limiting, and batch processing
        automatically. Texts are checked against the cache first, then
        uncached texts are sent to the provider in batches.

        Args:
            texts: List of text strings to embed
            use_cache: Whether to use cache (default: True)

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            ValueError: If texts is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        self._stats["total_requests"] += 1

        # Check cache first
        embeddings = []
        texts_to_embed = []
        cached_flags = []
        text_indices = []  # Track original positions

        if use_cache and self.cache:
            for idx, text in enumerate(texts):
                cache_key = self._compute_cache_key(text)
                cached_embedding = self.cache.get(cache_key)

                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    cached_flags.append(True)
                    self._stats["cache_hits"] += 1
                else:
                    texts_to_embed.append(text)
                    text_indices.append(idx)
                    cached_flags.append(False)
                    self._stats["cache_misses"] += 1
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
            cached_flags = [False] * len(texts)

        # Generate embeddings for uncached texts
        new_embeddings = []
        total_tokens = 0
        total_cost = 0.0

        if texts_to_embed:
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            # Split into batches if needed
            batches = self._create_batches(texts_to_embed)

            for batch in batches:
                try:
                    batch_result = self.provider.get_embeddings(batch)
                    new_embeddings.extend(batch_result.embeddings)

                    # Update stats
                    total_tokens += batch_result.token_count
                    total_cost += batch_result.cost

                    # Cache new embeddings
                    if self.cache:
                        for text, embedding in zip(
                            batch, batch_result.embeddings
                        ):
                            cache_key = self._compute_cache_key(text)
                            self.cache.set(cache_key, embedding)

                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    raise

            self._stats["total_tokens"] += total_tokens
            self._stats["total_cost"] += total_cost

        # Merge cached and new embeddings in correct order
        result_embeddings = [None] * len(texts)
        cached_idx = 0
        new_idx = 0

        for idx, is_cached in enumerate(cached_flags):
            if is_cached:
                result_embeddings[idx] = embeddings[cached_idx]
                cached_idx += 1
            else:
                result_embeddings[idx] = new_embeddings[new_idx]
                new_idx += 1

        # Create result
        return EmbeddingResult(
            embeddings=result_embeddings,
            model=self.provider.get_model_name(),
            dimensions=self.provider.get_dimensions(),
            token_count=total_tokens,
            cost=total_cost,
            provider=self.config.provider,
            cached=cached_flags
        )

    def _compute_cache_key(self, text: str) -> str:
        """Compute cache key from text.

        The cache key includes the model name to ensure different models
        don't share cache entries.

        Args:
            text: Text to compute key for

        Returns:
            SHA-256 hash of model name and text
        """
        model_name = self.provider.get_model_name()
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches based on provider limits.

        Args:
            texts: List of texts to batch

        Returns:
            List of text batches
        """
        max_batch_size = self.provider.get_max_batch_size()
        batches = []

        for i in range(0, len(texts), max_batch_size):
            batches.append(texts[i:i + max_batch_size])

        return batches

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with statistics including requests, cache hits/misses,
            tokens, cost, and cache hit rate
        """
        total_cache_requests = (
            self._stats["cache_hits"] + self._stats["cache_misses"]
        )
        cache_hit_rate = 0.0
        if total_cache_requests > 0:
            cache_hit_rate = self._stats["cache_hits"] / total_cache_requests

        stats = {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "model": self.provider.get_model_name(),
            "provider": self.config.provider
        }

        # Add cache stats if available
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    def clear_cache(self):
        """Clear the embedding cache.

        Does nothing if caching is not enabled.
        """
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
