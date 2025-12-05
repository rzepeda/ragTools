"""LLM service configuration."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMServiceConfig(BaseModel):
    """Configuration for LLM service.

    Attributes:
        provider: LLM provider name (anthropic, openai, ollama)
        model: Model name to use
        provider_config: Provider-specific configuration
        enable_rate_limiting: Whether to enable rate limiting
        rate_limit_config: Rate limiter configuration
    """

    provider: str = Field(
        "anthropic",
        description="LLM provider name"
    )
    model: Optional[str] = Field(
        None,
        description="Model name to use"
    )
    provider_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration"
    )
    enable_rate_limiting: bool = Field(
        True,
        description="Enable rate limiting"
    )
    rate_limit_config: Dict[str, Any] = Field(
        default_factory=lambda: {"requests_per_minute": 50},
        description="Rate limiter configuration"
    )

    def __init__(self, **data):
        """Initialize config with environment variable support."""
        super().__init__(**data)

        # Auto-populate API keys from environment if not provided
        if self.provider == "anthropic" and "api_key" not in self.provider_config:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.provider_config["api_key"] = api_key

        if self.provider == "openai" and "api_key" not in self.provider_config:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.provider_config["api_key"] = api_key

        # Set default model for provider if not specified
        if self.model:
            self.provider_config["model"] = self.model

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
