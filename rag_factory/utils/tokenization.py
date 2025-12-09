"""
Tokenization utilities using tiktoken.

This module provides lightweight tokenization using tiktoken instead of
the heavy transformers library. It includes fallback tokenization for
cases where tiktoken is unavailable.
"""

from typing import List, Optional
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenizationError(Exception):
    """Raised when tokenization fails."""
    pass


@lru_cache(maxsize=10)
def get_encoding(encoding_name: str = "cl100k_base"):
    """
    Get tiktoken encoding with caching.

    Args:
        encoding_name: Name of the encoding (cl100k_base, p50k_base, r50k_base)

    Returns:
        Tiktoken encoding

    Raises:
        TokenizationError: If encoding cannot be loaded
    """
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        raise TokenizationError(f"Failed to load encoding '{encoding_name}': {e}")


def get_encoding_for_model(model_name: str):
    """
    Get appropriate encoding for a model.

    Args:
        model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        Tiktoken encoding
    """
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Default to cl100k_base for unknown models
        logger.warning(
            f"No specific encoding for model '{model_name}', "
            f"using cl100k_base"
        )
        return get_encoding("cl100k_base")
    except Exception as e:
        raise TokenizationError(f"Failed to get encoding for model '{model_name}': {e}")


class Tokenizer:
    """
    Lightweight tokenizer using tiktoken.
    Falls back to basic tokenization if tiktoken unavailable.
    """

    def __init__(
        self,
        encoding_name: Optional[str] = None,
        model_name: Optional[str] = None,
        use_fallback: bool = True
    ):
        """
        Initialize tokenizer.

        Args:
            encoding_name: Tiktoken encoding name (cl100k_base, p50k_base, r50k_base)
            model_name: Model name (alternative to encoding_name)
            use_fallback: Whether to use fallback tokenization if tiktoken fails
        """
        self.use_fallback = use_fallback
        self.encoding = None
        self.encoding_name = None

        try:
            if model_name:
                self.encoding = get_encoding_for_model(model_name)
                self.encoding_name = model_name
            elif encoding_name:
                self.encoding = get_encoding(encoding_name)
                self.encoding_name = encoding_name
            else:
                # Default to cl100k_base (GPT-4, GPT-3.5-turbo)
                self.encoding = get_encoding("cl100k_base")
                self.encoding_name = "cl100k_base"

            logger.info(f"Initialized tokenizer with encoding: {self.encoding_name}")

        except Exception as e:
            if use_fallback:
                logger.warning(
                    f"Failed to initialize tiktoken ({e}), "
                    f"falling back to basic tokenization"
                )
                self.encoding = None
                self.encoding_name = "fallback"
            else:
                raise TokenizationError(f"Failed to initialize tokenizer: {e}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if self.encoding:
            return self.encoding.encode(text)
        else:
            return self._fallback_encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if self.encoding:
            return self.encoding.decode(tokens)
        else:
            return self._fallback_decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    def truncate(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to maximum token count.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        tokens = self.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Reserve tokens for suffix
        suffix_tokens = self.encode(suffix)
        max_content_tokens = max_tokens - len(suffix_tokens)

        if max_content_tokens <= 0:
            return suffix

        truncated_tokens = tokens[:max_content_tokens]
        truncated_text = self.decode(truncated_tokens)

        return truncated_text + suffix

    def split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks by token count.

        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        tokens = self.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move to next chunk with overlap
            if end >= len(tokens):
                break
            start = end - overlap

        return chunks

    def _fallback_encode(self, text: str) -> List[int]:
        """
        Basic fallback tokenization (word-based).

        Args:
            text: Text to tokenize

        Returns:
            List of pseudo token IDs
        """
        # Simple word-based tokenization
        words = re.findall(r'\w+|[^\w\s]', text)
        # Use hash of word as token ID
        return [hash(word) % 100000 for word in words]

    def _fallback_decode(self, tokens: List[int]) -> str:
        """
        Basic fallback detokenization.

        Note: This is lossy and only for compatibility.
        """
        logger.warning("Fallback decode is lossy and approximate")
        return " ".join(f"<token_{t}>" for t in tokens)


# Convenience functions
def count_tokens(
    text: str,
    model: str = "gpt-3.5-turbo"
) -> int:
    """
    Count tokens in text for a specific model.

    Args:
        text: Text to count tokens in
        model: Model name

    Returns:
        Number of tokens
    """
    tokenizer = Tokenizer(model_name=model)
    return tokenizer.count_tokens(text)


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = "gpt-3.5-turbo",
    suffix: str = "..."
) -> str:
    """
    Truncate text to maximum tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        model: Model name
        suffix: Truncation suffix

    Returns:
        Truncated text
    """
    tokenizer = Tokenizer(model_name=model)
    return tokenizer.truncate(text, max_tokens, suffix)


def split_text_by_tokens(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Split text into chunks by token count.

    Args:
        text: Text to split
        chunk_size: Maximum tokens per chunk
        overlap: Overlapping tokens
        model: Model name

    Returns:
        List of text chunks
    """
    tokenizer = Tokenizer(model_name=model)
    return tokenizer.split_by_tokens(text, chunk_size, overlap)
