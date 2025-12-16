"""Configuration for embedding service."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class EmbeddingServiceConfig:
    """Configuration for the embedding service.

    Attributes:
        provider: Name of the embedding provider (openai, cohere, local)
        model: Model name to use for embeddings
        provider_config: Provider-specific configuration
        enable_cache: Whether to enable embedding caching
        cache_config: Cache configuration
        enable_rate_limiting: Whether to enable rate limiting
        rate_limit_config: Rate limiting configuration
    """
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    provider_config: Dict[str, Any] = field(default_factory=dict)
    enable_cache: bool = True
    cache_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_size": 10000,
        "ttl": 3600
    })
    enable_rate_limiting: bool = False
    rate_limit_config: Dict[str, Any] = field(default_factory=lambda: {
        "requests_per_minute": 3000
    })

    def __post_init__(self):
        """Post-initialization to set defaults and validate."""
        # Set provider config defaults
        if not self.provider_config:
            self.provider_config = self._get_default_provider_config()

        # Validate provider
        valid_providers = ["openai", "cohere", "local", "onnx-local"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. "
                f"Must be one of {valid_providers}"
            )

    def _get_default_provider_config(self) -> Dict[str, Any]:
        """Get default provider configuration based on provider type."""
        if self.provider == "openai":
            return {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": self.model,
                "max_batch_size": 100
            }
        elif self.provider == "cohere":
            return {
                "api_key": os.getenv("COHERE_API_KEY", ""),
                "model": self.model,
                "max_batch_size": 96
            }
        elif self.provider == "local":
            return {
                "model": self.model,
                "max_batch_size": 32,
                "device": "cpu"
            }
        return {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmbeddingServiceConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EmbeddingServiceConfig instance
        """
        return cls(**config_dict)
