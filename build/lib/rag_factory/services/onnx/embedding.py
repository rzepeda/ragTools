"""ONNX-based embedding service implementation.

This module provides an embedding service that implements IEmbeddingService
by wrapping the existing ONNXLocalProvider.
"""

from typing import List
import asyncio
from pathlib import Path

from rag_factory.services.interfaces import IEmbeddingService
from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider


class ONNXEmbeddingService(IEmbeddingService):
    """ONNX-based local embedding service.

    This service implements IEmbeddingService using ONNX Runtime for
    lightweight, local embedding generation without PyTorch dependencies.

    Example:
        >>> service = ONNXEmbeddingService(
        ...     model_path="/path/to/model.onnx",
        ...     tokenizer_path="/path/to/tokenizer"
        ... )
        >>> embedding = await service.embed("Hello world")
        >>> print(len(embedding))  # e.g., 384
        >>> embeddings = await service.embed_batch(["Hello", "World"])
        >>> print(len(embeddings))  # 2
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_path: str = None,
        cache_dir: str = None,
        max_batch_size: int = 32,
        max_length: int = 512,
        num_threads: int = None,
    ):
        """Initialize ONNX embedding service.

        Args:
            model: Model name from HuggingFace (default: all-MiniLM-L6-v2)
            model_path: Optional path to local ONNX model file
            cache_dir: Optional directory for model cache
            max_batch_size: Maximum batch size for processing (default: 32)
            max_length: Maximum sequence length (default: 512)
            num_threads: Number of CPU threads for inference (optional)

        Raises:
            ImportError: If ONNX Runtime is not installed
        """
        config = {
            "model": model,
            "max_batch_size": max_batch_size,
            "max_length": max_length,
        }

        if model_path:
            config["model_path"] = model_path
        if cache_dir:
            config["cache_dir"] = cache_dir
        if num_threads:
            config["num_threads"] = num_threads

        # Initialize the underlying ONNX provider
        self._provider = ONNXLocalProvider(config)

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
