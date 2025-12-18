"""Main LLM service implementation."""

from typing import List, Dict, Any, Optional, Iterator, AsyncIterator, Callable
import time
import logging

from .base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole
from .config import LLMServiceConfig
from ..embedding.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class LLMService:
    """Centralized LLM service with multi-provider support.

    Example:
        config = LLMServiceConfig(provider="anthropic", model="claude-sonnet-4.5")
        service = LLMService(config)

        messages = [Message(role=MessageRole.USER, content="Hello!")]
        response = service.complete(messages)
        print(response.content)
    """

    def __init__(self, config: LLMServiceConfig):
        """Initialize LLM service.

        Args:
            config: LLM service configuration
        """
        self.config = config
        self.provider = self._init_provider()
        self.rate_limiter = (
            RateLimiter(config.rate_limit_config)
            if config.enable_rate_limiting
            else None
        )
        self._stats = {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
        }

    def _init_provider(self) -> ILLMProvider:
        """Initialize LLM provider based on config.

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider is unknown
        """
        from .providers.anthropic import AnthropicProvider
        from .providers.openai import OpenAIProvider
        from .providers.ollama import OllamaProvider

        provider_map = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
        }

        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return provider_class(self.config.provider_config)

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion for conversation.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ValueError: If messages is empty
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        self._stats["total_requests"] += 1

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Count tokens before sending
        prompt_tokens = self.provider.count_tokens(messages)
        logger.info(f"Sending request with {prompt_tokens} prompt tokens")

        # Generate completion
        start = time.time()
        try:
            response = self.provider.complete(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

            # Update stats
            self._stats["total_prompt_tokens"] += response.prompt_tokens
            self._stats["total_completion_tokens"] += response.completion_tokens
            self._stats["total_cost"] += response.cost
            self._stats["total_latency"] += response.latency

            logger.info(
                f"Completed in {response.latency:.2f}s, "
                f"{response.total_tokens} tokens, "
                f"${response.cost:.6f}"
            )

            return response

        except Exception as e:
            logger.error(f"Error in LLM completion: {e}")
            raise

    async def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each chunk
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk for each piece of generated content

        Raises:
            ValueError: If messages is empty
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        self._stats["total_requests"] += 1

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Stream completion
        try:
            async for chunk in self.provider.stream(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            ):
                if callback:
                    callback(chunk.content)

                yield chunk

        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            raise

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages.

        Args:
            messages: List of messages

        Returns:
            Number of tokens
        """
        return self.provider.count_tokens(messages)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        avg_latency = 0.0
        if self._stats["total_requests"] > 0:
            avg_latency = self._stats["total_latency"] / self._stats["total_requests"]

        return {
            **self._stats,
            "average_latency": avg_latency,
            "model": self.provider.get_model_name(),
            "provider": self.config.provider,
        }

    def estimate_cost(self, messages: List[Message], max_completion_tokens: int) -> float:
        """Estimate cost for a request.

        Args:
            messages: List of messages
            max_completion_tokens: Expected completion tokens

        Returns:
            Estimated cost in dollars
        """
        prompt_tokens = self.provider.count_tokens(messages)
        return self.provider.calculate_cost(prompt_tokens, max_completion_tokens)
