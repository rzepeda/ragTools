"""
Integration tests for tokenization.
"""

import pytest
from rag_factory.utils.tokenization import Tokenizer, count_tokens


@pytest.mark.integration
class TestTokenizationIntegration:
    """Integration tests for tokenization."""

    def test_openai_token_count_accuracy(self):
        """Test that token counts match OpenAI's counts."""
        # Known token counts from OpenAI for gpt-3.5-turbo
        # These are approximate and may vary slightly
        test_cases = [
            ("Hello, world!", 4),
            ("The quick brown fox jumps over the lazy dog.", 10),
            ("This is a test.", 5),
        ]

        tokenizer = Tokenizer(model_name="gpt-3.5-turbo")

        for text, expected_count in test_cases:
            actual_count = tokenizer.count_tokens(text)
            # Allow small variance (±1 token)
            assert abs(actual_count - expected_count) <= 1, \
                f"Token count mismatch for '{text}': expected ~{expected_count}, got {actual_count}"

    def test_different_encodings(self):
        """Test different tiktoken encodings."""
        text = "This is a test sentence."

        encodings = ["cl100k_base", "p50k_base", "r50k_base"]

        counts = {}
        for encoding in encodings:
            tokenizer = Tokenizer(encoding_name=encoding)
            counts[encoding] = tokenizer.count_tokens(text)

        # Different encodings should give different counts (usually)
        # At minimum, all should be positive
        assert all(count > 0 for count in counts.values())

    def test_long_text_handling(self):
        """Test handling of very long texts."""
        # Create a long text
        long_text = "This is a sentence. " * 1000

        tokenizer = Tokenizer()

        # Should handle without errors
        tokens = tokenizer.encode(long_text)
        assert len(tokens) > 1000

        # Should be able to decode
        decoded = tokenizer.decode(tokens)
        assert decoded == long_text

    def test_special_characters(self):
        """Test handling of special characters."""
        texts = [
            "Hello 👋 World 🌍",
            "Code: `print('hello')`",
            "Math: x² + y² = z²",
            "Symbols: @#$%^&*()",
            "Quotes: \"double\" and 'single'",
            "Newlines:\nand\ttabs",
        ]

        tokenizer = Tokenizer()

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed to round-trip: {text}"

    def test_multilingual_text(self):
        """Test handling of multilingual text."""
        texts = [
            "Hello",  # English
            "Bonjour",  # French
            "Hola",  # Spanish
            "こんにちは",  # Japanese
            "你好",  # Chinese
            "مرحبا",  # Arabic
            "Привет",  # Russian
            "안녕하세요",  # Korean
        ]

        tokenizer = Tokenizer()

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for language: {text}"
            assert len(tokens) > 0

    def test_mixed_content(self):
        """Test text with mixed content types."""
        text = """
        # Heading
        
        This is a paragraph with **bold** and *italic* text.
        
        ```python
        def hello():
            print("Hello, world! 👋")
        ```
        
        - List item 1
        - List item 2
        
        Math: E = mc²
        """

        tokenizer = Tokenizer()

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text
        assert len(tokens) > 20

    def test_truncation_accuracy(self):
        """Test that truncation respects token limits."""
        tokenizer = Tokenizer()
        text = "This is a longer text that will be truncated to a specific token count."

        for max_tokens in [5, 10, 15, 20]:
            truncated = tokenizer.truncate(text, max_tokens=max_tokens)
            token_count = tokenizer.count_tokens(truncated)
            
            # Should be at or under the limit (accounting for suffix)
            assert token_count <= max_tokens + 2, \
                f"Truncated text has {token_count} tokens, expected <={max_tokens + 2}"

    def test_splitting_accuracy(self):
        """Test that text splitting respects chunk sizes."""
        tokenizer = Tokenizer()
        text = "This is a longer text. " * 50  # Create a longer text

        chunk_size = 20
        chunks = tokenizer.split_by_tokens(text, chunk_size=chunk_size, overlap=0)

        # All chunks except possibly the last should be at the limit
        for i, chunk in enumerate(chunks[:-1]):
            token_count = tokenizer.count_tokens(chunk)
            assert token_count <= chunk_size, \
                f"Chunk {i} has {token_count} tokens, expected <={chunk_size}"

        # Last chunk can be smaller
        last_chunk_count = tokenizer.count_tokens(chunks[-1])
        assert last_chunk_count <= chunk_size

    def test_overlap_behavior(self):
        """Test that overlap works correctly."""
        tokenizer = Tokenizer()
        text = "Word " * 100  # Simple repeated text

        chunk_size = 20
        overlap = 5

        chunks = tokenizer.split_by_tokens(text, chunk_size=chunk_size, overlap=overlap)

        # With overlap, we should have more chunks than without
        chunks_no_overlap = tokenizer.split_by_tokens(text, chunk_size=chunk_size, overlap=0)
        assert len(chunks) >= len(chunks_no_overlap)

    def test_empty_and_whitespace(self):
        """Test handling of empty and whitespace-only text."""
        tokenizer = Tokenizer()

        # Empty text
        assert tokenizer.count_tokens("") == 0
        assert tokenizer.encode("") == []

        # Whitespace only
        whitespace_count = tokenizer.count_tokens("   ")
        assert whitespace_count >= 0  # May be 0 or 1 depending on encoding

    def test_very_long_single_word(self):
        """Test handling of very long single words."""
        tokenizer = Tokenizer()
        
        # Create a very long "word"
        long_word = "a" * 1000
        
        tokens = tokenizer.encode(long_word)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == long_word
        assert len(tokens) > 0

    def test_consistency_across_calls(self):
        """Test that tokenization is consistent across multiple calls."""
        tokenizer = Tokenizer()
        text = "This is a test sentence for consistency checking."

        # Encode multiple times
        tokens1 = tokenizer.encode(text)
        tokens2 = tokenizer.encode(text)
        tokens3 = tokenizer.encode(text)

        # Should be identical
        assert tokens1 == tokens2 == tokens3

    def test_different_models(self):
        """Test tokenization with different model names."""
        text = "This is a test sentence."

        models = ["gpt-4", "gpt-3.5-turbo", "gpt-3"]

        counts = {}
        for model in models:
            try:
                tokenizer = Tokenizer(model_name=model)
                counts[model] = tokenizer.count_tokens(text)
            except Exception as e:
                pytest.fail(f"Failed to initialize tokenizer for {model}: {e}")

        # All should produce valid counts
        assert all(count > 0 for count in counts.values())


@pytest.mark.integration
class TestPerformance:
    """Performance tests for tokenization."""

    def test_tokenization_speed(self):
        """Test that tokenization is fast enough."""
        import time
        
        tokenizer = Tokenizer()
        text = "This is a test sentence. " * 50  # ~512 tokens

        start = time.time()
        for _ in range(100):
            tokenizer.encode(text)
        end = time.time()

        avg_time = (end - start) / 100
        
        # Should be fast (< 10ms per document as per requirements)
        assert avg_time < 0.01, f"Tokenization too slow: {avg_time*1000:.2f}ms"

    def test_token_counting_speed(self):
        """Test that token counting is fast."""
        import time
        
        text = "This is a query."

        start = time.time()
        for _ in range(1000):
            count_tokens(text)
        end = time.time()

        avg_time = (end - start) / 1000
        
        # Should be very fast (< 1ms as per requirements)
        assert avg_time < 0.001, f"Token counting too slow: {avg_time*1000:.2f}ms"

    def test_batch_tokenization_performance(self):
        """Test performance with batch tokenization."""
        import time
        
        tokenizer = Tokenizer()
        texts = ["This is a test sentence. " * 10] * 100

        start = time.time()
        for text in texts:
            tokenizer.encode(text)
        end = time.time()

        total_time = end - start
        
        # Should handle 100 documents in reasonable time
        assert total_time < 1.0, f"Batch tokenization too slow: {total_time:.2f}s"
