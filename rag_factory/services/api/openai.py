"""OpenAI LLM and Embedding service implementations.

This module provides LLM and embedding services that implement ILLMService
and IEmbeddingService by wrapping the existing OpenAIProvider.
"""

from typing import AsyncIterator, Optional, Any, List
import asyncio

from rag_factory.services.interfaces import ILLMService, IEmbeddingService
from rag_factory.services.llm.providers.openai import OpenAIProvider as OpenAILLMProvider
from rag_factory.services.embedding.providers.openai import OpenAIProvider as OpenAIEmbeddingProvider
from rag_factory.services.llm.base import Message, MessageRole


class OpenAILLMService(ILLMService):
    """OpenAI GPT API service.

    This service implements ILLMService using OpenAI's GPT models
    for text generation.

    Example:
        >>> service = OpenAILLMService(
        ...     api_key="your-api-key",
        ...     model="gpt-4"
        ... )
        >>> response = await service.complete("What is RAG?")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4"
    ):
        """Initialize OpenAI LLM service.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4)

        Raises:
            ValueError: If model is not supported
        """
        config = {
            "api_key": api_key,
            "model": model,
        }

        # Initialize the underlying OpenAI provider
        self._provider = OpenAILLMProvider(config)

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

        # Stream from provider
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


class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI Embedding API service.

    This service implements IEmbeddingService using OpenAI's embedding models.

    Example:
        >>> service = OpenAIEmbeddingService(
        ...     api_key="your-api-key",
        ...     model="text-embedding-3-small"
        ... )
        >>> embedding = await service.embed("Hello world")
        >>> print(len(embedding))  # 1536
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):
        """Initialize OpenAI embedding service.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
        """
        config = {
            "api_key": api_key,
            "model": model,
        }

        # Initialize the underlying OpenAI embedding provider
        self._provider = OpenAIEmbeddingProvider(config)

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        # Run sync provider in executor to make it async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._provider.get_embeddings,
            [text]
        )

        # Extract first embedding from result
        if result.embeddings:
            return result.embeddings[0]
        return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        This method is more efficient than calling embed() multiple times
        as it can batch requests to the underlying provider.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return []

        # Run sync provider in executor to make it async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._provider.get_embeddings,
            texts
        )

        return result.embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Number of dimensions in the embedding vectors
        """
        return self._provider.get_dimensions()
