"""
Unit tests for tokenization utilities.
"""

import pytest
from rag_factory.utils.tokenization import (
    Tokenizer,
    TokenizationError,
    get_encoding,
    get_encoding_for_model,
    count_tokens,
    truncate_text,
    split_text_by_tokens,
)


class TestTokenizer:
    """Test tokenizer functionality."""

    def test_initialization_with_encoding(self):
        """Test tokenizer initialization with encoding name."""
        tokenizer = Tokenizer(encoding_name="cl100k_base")
        assert tokenizer.encoding is not None
        assert tokenizer.encoding_name == "cl100k_base"

    def test_initialization_with_model(self):
        """Test tokenizer initialization with model name."""
        tokenizer = Tokenizer(model_name="gpt-4")
        assert tokenizer.encoding is not None
        assert tokenizer.encoding_name == "gpt-4"

    def test_default_initialization(self):
        """Test tokenizer initialization with defaults."""
        tokenizer = Tokenizer()
        assert tokenizer.encoding is not None
        assert tokenizer.encoding_name == "cl100k_base"

    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = Tokenizer()
        text = "Hello, world!"

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert decoded == text

    def test_count_tokens(self):
        """Test token counting."""
        tokenizer = Tokenizer()
        text = "This is a test sentence."

        count = tokenizer.count_tokens(text)

        assert count > 0
        assert count == len(tokenizer.encode(text))

    def test_truncate(self):
        """Test text truncation."""
        tokenizer = Tokenizer()
        text = "This is a long text that needs to be truncated."

        truncated = tokenizer.truncate(text, max_tokens=5)

        # Allow some tolerance for suffix tokens
        assert tokenizer.count_tokens(truncated) <= 7
        assert truncated.endswith("...")

    def test_truncate_no_truncation_needed(self):
        """Test truncation when text is short enough."""
        tokenizer = Tokenizer()
        text = "Short text"

        truncated = tokenizer.truncate(text, max_tokens=100)

        assert truncated == text

    def test_split_by_tokens(self):
        """Test splitting text by tokens."""
        tokenizer = Tokenizer()
        text = "This is a longer text that will be split into multiple chunks."

        chunks = tokenizer.split_by_tokens(text, chunk_size=5, overlap=0)

        assert len(chunks) > 1
        for chunk in chunks[:-1]:  # All but last should be at limit
            assert tokenizer.count_tokens(chunk) <= 5

    def test_split_with_overlap(self):
        """Test splitting with overlap."""
        tokenizer = Tokenizer()
        text = "This is a test sentence for overlap testing."

        chunks_no_overlap = tokenizer.split_by_tokens(text, chunk_size=5, overlap=0)
        chunks_with_overlap = tokenizer.split_by_tokens(text, chunk_size=5, overlap=2)

        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        tokenizer = Tokenizer()
        text = "Short"

        chunks = tokenizer.split_by_tokens(text, chunk_size=100, overlap=0)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_fallback_tokenization(self):
        """Test fallback tokenization when tiktoken unavailable."""
        # Force fallback by using invalid encoding
        tokenizer = Tokenizer(encoding_name="invalid_encoding", use_fallback=True)

        text = "Test text"
        tokens = tokenizer.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert tokenizer.encoding_name == "fallback"

    def test_no_fallback_raises_error(self):
        """Test that invalid encoding raises error when fallback disabled."""
        with pytest.raises(TokenizationError):
            Tokenizer(encoding_name="invalid_encoding", use_fallback=False)

    def test_different_encodings(self):
        """Test different tiktoken encodings."""
        encodings = ["cl100k_base", "p50k_base", "r50k_base"]
        text = "This is a test sentence."

        counts = {}
        for encoding in encodings:
            tokenizer = Tokenizer(encoding_name=encoding)
            counts[encoding] = tokenizer.count_tokens(text)

        # Different encodings may give different counts
        assert all(count > 0 for count in counts.values())

    def test_special_characters(self):
        """Test handling of special characters."""
        tokenizer = Tokenizer()
        texts = [
            "Hello 👋 World 🌍",
            "Code: `print('hello')`",
            "Math: x² + y² = z²",
            "Symbols: @#$%^&*()",
        ]

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text

    def test_empty_text(self):
        """Test handling of empty text."""
        tokenizer = Tokenizer()
        
        tokens = tokenizer.encode("")
        assert len(tokens) == 0
        
        count = tokenizer.count_tokens("")
        assert count == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_count_tokens(self):
        """Test count_tokens convenience function."""
        text = "Test sentence"
        count = count_tokens(text, model="gpt-3.5-turbo")
        assert count > 0

    def test_truncate_text(self):
        """Test truncate_text convenience function."""
        text = "This is a long text that needs truncation"
        truncated = truncate_text(text, max_tokens=3, model="gpt-3.5-turbo")
        assert "..." in truncated

    def test_split_text_by_tokens(self):
        """Test split_text_by_tokens convenience function."""
        text = "This is a longer text for splitting"
        chunks = split_text_by_tokens(text, chunk_size=3, model="gpt-3.5-turbo")
        assert len(chunks) > 0


class TestEncodingFunctions:
    """Test encoding getter functions."""

    def test_get_encoding(self):
        """Test get_encoding function."""
        encoding = get_encoding("cl100k_base")
        assert encoding is not None

    def test_get_encoding_caching(self):
        """Test that get_encoding uses caching."""
        encoding1 = get_encoding("cl100k_base")
        encoding2 = get_encoding("cl100k_base")
        # Should be the same object due to caching
        assert encoding1 is encoding2

    def test_get_encoding_invalid(self):
        """Test get_encoding with invalid encoding."""
        with pytest.raises(TokenizationError):
            get_encoding("invalid_encoding")

    def test_get_encoding_for_model(self):
        """Test get_encoding_for_model function."""
        encoding = get_encoding_for_model("gpt-4")
        assert encoding is not None

    def test_get_encoding_for_unknown_model(self):
        """Test get_encoding_for_model with unknown model."""
        # Should fall back to cl100k_base
        encoding = get_encoding_for_model("unknown-model-xyz")
        assert encoding is not None


class TestMultilingualSupport:
    """Test multilingual text handling."""

    def test_multilingual_text(self):
        """Test handling of multilingual text."""
        tokenizer = Tokenizer()
        texts = [
            "Hello",  # English
            "Bonjour",  # French
            "こんにちは",  # Japanese
            "你好",  # Chinese
            "مرحبا",  # Arabic
        ]

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
            assert len(tokens) > 0

    def test_mixed_language_text(self):
        """Test text with mixed languages."""
        tokenizer = Tokenizer()
        text = "Hello 你好 Bonjour こんにちは"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
        assert len(tokens) > 0
