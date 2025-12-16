"""Anthropic Claude LLM service implementation.

This module provides an LLM service that implements ILLMService
by wrapping the existing AnthropicProvider.
"""

from typing import AsyncIterator, Optional, Any
import asyncio

from rag_factory.services.interfaces import ILLMService
from rag_factory.services.llm.providers.anthropic import AnthropicProvider
from rag_factory.services.llm.base import Message, MessageRole


class AnthropicLLMService(ILLMService):
    """Anthropic Claude API service.

    This service implements ILLMService using Anthropic's Claude models
    for text generation.

    Example:
        >>> service = AnthropicLLMService(
        ...     api_key="your-api-key",
        ...     model="claude-sonnet-4.5"
        ... )
        >>> response = await service.complete("What is RAG?")
        >>> print(response)
        >>> async for chunk in service.stream_complete("Tell me a story"):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4.5"
    ):
        """Initialize Anthropic LLM service.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-sonnet-4.5)

        Raises:
            ValueError: If model is not supported
        """
        config = {
            "api_key": api_key,
            "model": model,
        }

        # Initialize the underlying Anthropic provider
        self._provider = AnthropicProvider(config)

    async def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate completion for prompt.

        Args:
            prompt: Input text prompt for generation
            max_tokens: Maximum number of tokens to generate (provider-specific default if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text completion

        Raises:
            Exception: If completion generation fails
        """
        # Convert prompt to message format
        messages = [Message(role=MessageRole.USER, content=prompt)]

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = 1000

        # Run sync provider in executor to make it async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        )

        return response.content

    async def stream_complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            prompt: Input text prompt for generation
            max_tokens: Maximum number of tokens to generate (provider-specific default if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional provider-specific parameters

        Yields:
            Generated text tokens as they are produced

        Raises:
            Exception: If streaming fails
        """
        # Convert prompt to message format
        messages = [Message(role=MessageRole.USER, content=prompt)]

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = 1000

        # Stream from provider (it's a sync generator, so we need to wrap it)
        loop = asyncio.get_event_loop()

        def _stream():
            return self._provider.stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        # Get the generator
        stream_gen = await loop.run_in_executor(None, _stream)

        # Yield chunks
        for chunk in stream_gen:
            if not chunk.is_final:
                yield chunk.content
