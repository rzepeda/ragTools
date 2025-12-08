# Story 10.2: Replace Tokenization with Tiktoken

**Story ID:** 10.2
**Epic:** Epic 10 - Lightweight Dependencies Implementation
**Story Points:** 5
**Priority:** High
**Dependencies:** None (can run in parallel with 10.1)

---

## User Story

**As a** developer
**I want** to use tiktoken for tokenization
**So that** I avoid the heavy transformers library dependency

---

## Detailed Requirements

### Functional Requirements

1. **Tiktoken Integration**
   - Integrate tiktoken library for tokenization
   - Support multiple tiktoken encodings (cl100k_base, p50k_base, r50k_base, p50k_edit)
   - Implement encoding selection based on model type
   - Provide automatic encoding detection for common models
   - Handle encoding errors gracefully
   - Cache encodings for performance

2. **Transformers Replacement**
   - Replace all transformers tokenizer usage with tiktoken
   - Remove transformers dependency from requirements.txt
   - Update all strategies using tokenization
   - Ensure backward compatibility where possible
   - Add migration guide for existing code

3. **Token Counting**
   - Implement accurate token counting for various models
   - Support token counting for OpenAI models (GPT-3.5, GPT-4)
   - Support token counting for other common models
   - Provide token limit validation
   - Add token budget tracking

4. **Fallback Tokenization**
   - Implement basic regex-based tokenization as fallback
   - Use fallback when tiktoken is unavailable
   - Provide clear warnings when using fallback
   - Document fallback behavior and limitations
   - Ensure fallback is "good enough" for basic use cases

5. **Tokenization Utilities**
   - Create tokenization utility module
   - Implement text chunking by token count
   - Add token-aware text splitting
   - Provide encoding/decoding utilities
   - Support special tokens handling

6. **Strategy Updates**
   - Update Late Chunking strategy to use tiktoken
   - Update Contextual RAG strategy to use tiktoken
   - Update any other strategies using tokenization
   - Ensure all token counting is accurate
   - Validate performance after migration

### Non-Functional Requirements

1. **Performance**
   - Tokenization speed: <10ms for typical documents (512 tokens)
   - Token counting: <1ms for queries
   - Encoding caching to avoid repeated loads
   - Minimal memory overhead

2. **Reliability**
   - Handle encoding errors gracefully
   - Provide clear error messages
   - Fallback to basic tokenization if tiktoken fails
   - Validate token counts against known benchmarks

3. **Compatibility**
   - Support Python 3.8+
   - Work on all platforms (Linux, macOS, Windows)
   - Compatible with all supported models
   - Backward compatible API where possible

4. **Maintainability**
   - Clear tokenization interface
   - Well-documented encoding selection
   - Comprehensive error handling
   - Extensive logging for debugging

5. **Resource Efficiency**
   - tiktoken library: ~5MB (vs transformers ~500MB)
   - Fast loading time (<100ms)
   - Low memory footprint
   - No external model downloads required

---

## Acceptance Criteria

### AC1: Tiktoken Integration
- [ ] tiktoken library integrated
- [ ] Support for cl100k_base encoding (GPT-4, GPT-3.5-turbo)
- [ ] Support for p50k_base encoding (GPT-3)
- [ ] Support for r50k_base encoding (older models)
- [ ] Encoding caching implemented
- [ ] Automatic encoding selection working

### AC2: Transformers Removal
- [ ] All transformers tokenizer usage replaced
- [ ] transformers dependency removed from requirements.txt
- [ ] No transformers imports in core code
- [ ] Migration guide written
- [ ] Backward compatibility maintained where possible

### AC3: Token Counting
- [ ] Accurate token counting for OpenAI models
- [ ] Token counting for other common models
- [ ] Token limit validation working
- [ ] Token budget tracking implemented
- [ ] Token counts match OpenAI's counts (for OpenAI models)

### AC4: Fallback Tokenization
- [ ] Basic regex tokenization implemented
- [ ] Fallback activates when tiktoken unavailable
- [ ] Warning messages displayed when using fallback
- [ ] Fallback behavior documented
- [ ] Fallback tested and validated

### AC5: Utilities
- [ ] Tokenization utility module created
- [ ] Text chunking by token count working
- [ ] Token-aware text splitting implemented
- [ ] Encoding/decoding utilities available
- [ ] Special tokens handling working

### AC6: Strategy Updates
- [ ] Late Chunking updated to use tiktoken
- [ ] Contextual RAG updated to use tiktoken
- [ ] All other strategies updated
- [ ] Token counting validated in all strategies
- [ ] Performance maintained or improved

### AC7: Testing
- [ ] Unit tests for tokenization utilities
- [ ] Unit tests for encoding selection
- [ ] Integration tests with real text
- [ ] Token count validation tests
- [ ] Fallback behavior tests
- [ ] Performance benchmarks
- [ ] All tests passing without transformers

---

## Technical Specifications

### File Structure
```
rag_factory/
├── utils/
│   ├── __init__.py
│   ├── tokenization.py           # Main tokenization utilities
│   ├── token_counter.py          # Token counting
│   └── text_splitter.py          # Token-aware text splitting
│
├── strategies/
│   ├── late_chunking/
│   │   └── document_embedder.py  # Updated to use tiktoken
│   ├── contextual/
│   │   └── context_generator.py  # Updated to use tiktoken
│   └── ...

tests/
├── unit/
│   └── utils/
│       ├── test_tokenization.py
│       ├── test_token_counter.py
│       └── test_text_splitter.py
│
└── integration/
    └── test_tokenization_integration.py
```

### Dependencies
```python
# requirements.txt
tiktoken>=0.5.2                 # ~5MB - Fast tokenization

# REMOVED (saving ~500MB):
# transformers>=4.36.0          # ~500MB
```

### Tokenization Utilities
```python
# rag_factory/utils/tokenization.py
from typing import List, Optional, Union
import tiktoken
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenizationError(Exception):
    """Raised when tokenization fails."""
    pass


@lru_cache(maxsize=10)
def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """
    Get tiktoken encoding with caching.

    Args:
        encoding_name: Name of the encoding

    Returns:
        Tiktoken encoding

    Raises:
        TokenizationError: If encoding cannot be loaded
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        raise TokenizationError(f"Failed to load encoding '{encoding_name}': {e}")


def get_encoding_for_model(model_name: str) -> tiktoken.Encoding:
    """
    Get appropriate encoding for a model.

    Args:
        model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        Tiktoken encoding
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Default to cl100k_base for unknown models
        logger.warning(
            f"No specific encoding for model '{model_name}', "
            f"using cl100k_base"
        )
        return get_encoding("cl100k_base")


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
            encoding_name: Tiktoken encoding name
            model_name: Model name (alternative to encoding_name)
            use_fallback: Whether to use fallback tokenization
        """
        self.use_fallback = use_fallback
        self.encoding = None

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
            start = end - overlap
            if start >= len(tokens):
                break

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
```

### Token Counter
```python
# rag_factory/utils/token_counter.py
from typing import Dict, List, Optional
import logging
from .tokenization import Tokenizer

logger = logging.getLogger(__name__)


class TokenBudget:
    """
    Track token usage against a budget.
    """

    def __init__(
        self,
        max_tokens: int,
        model: str = "gpt-3.5-turbo",
        reserve: int = 0
    ):
        """
        Initialize token budget.

        Args:
            max_tokens: Maximum tokens allowed
            model: Model name for tokenization
            reserve: Reserved tokens (e.g., for response)
        """
        self.max_tokens = max_tokens
        self.reserve = reserve
        self.available = max_tokens - reserve
        self.used = 0
        self.tokenizer = Tokenizer(model_name=model)

    def add_text(self, text: str) -> bool:
        """
        Try to add text to budget.

        Args:
            text: Text to add

        Returns:
            True if text fits in budget, False otherwise
        """
        tokens = self.tokenizer.count_tokens(text)

        if self.used + tokens <= self.available:
            self.used += tokens
            return True
        else:
            logger.debug(
                f"Text ({tokens} tokens) exceeds budget "
                f"(used: {self.used}/{self.available})"
            )
            return False

    def add_texts(self, texts: List[str]) -> List[str]:
        """
        Add texts until budget is exhausted.

        Args:
            texts: List of texts to add

        Returns:
            List of texts that fit in budget
        """
        added = []

        for text in texts:
            if self.add_text(text):
                added.append(text)
            else:
                break

        return added

    def remaining(self) -> int:
        """Get remaining tokens in budget."""
        return self.available - self.used

    def reset(self):
        """Reset budget."""
        self.used = 0


class TokenCounter:
    """
    Count tokens for various purposes.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.

        Args:
            model: Model name for tokenization
        """
        self.tokenizer = Tokenizer(model_name=model)
        self.model = model

    def count_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> int:
        """
        Count tokens in chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count
        """
        # Based on OpenAI's token counting for chat
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        tokens = 0

        for message in messages:
            # Every message has overhead
            tokens += 4  # <im_start>{role}\n{content}<im_end>\n

            for key, value in message.items():
                tokens += self.tokenizer.count_tokens(str(value))

        tokens += 2  # Assistant reply priming

        return tokens

    def count_function_call(
        self,
        function_name: str,
        function_args: Dict[str, any]
    ) -> int:
        """
        Count tokens in a function call.

        Args:
            function_name: Function name
            function_args: Function arguments

        Returns:
            Token count
        """
        import json

        # Function call format
        call_str = json.dumps({
            "name": function_name,
            "arguments": function_args
        })

        return self.tokenizer.count_tokens(call_str)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/utils/test_tokenization.py
import pytest
from rag_factory.utils.tokenization import (
    Tokenizer,
    get_encoding,
    count_tokens,
    truncate_text,
    split_text_by_tokens
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

        assert tokenizer.count_tokens(truncated) <= 5 + 1  # +1 for suffix
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

        chunks = tokenizer.split_by_tokens(text, chunk_size=5, overlap=2)

        assert len(chunks) > 1
        # Verify overlap exists (simplified check)
        assert len(chunks) > len(tokenizer.split_by_tokens(text, chunk_size=5, overlap=0))

    def test_fallback_tokenization(self):
        """Test fallback tokenization when tiktoken unavailable."""
        # Force fallback by using invalid encoding
        tokenizer = Tokenizer(encoding_name="invalid_encoding", use_fallback=True)

        text = "Test text"
        tokens = tokenizer.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_convenience_functions(self):
        """Test convenience functions."""
        text = "Test sentence"

        # count_tokens
        count = count_tokens(text, model="gpt-3.5-turbo")
        assert count > 0

        # truncate_text
        truncated = truncate_text(text, max_tokens=2, model="gpt-3.5-turbo")
        assert "..." in truncated

        # split_text_by_tokens
        chunks = split_text_by_tokens(text, chunk_size=2, model="gpt-3.5-turbo")
        assert len(chunks) > 0


class TestTokenCounter:
    """Test token counter."""

    def test_count_messages(self):
        """Test counting tokens in messages."""
        from rag_factory.utils.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-3.5-turbo")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        count = counter.count_messages(messages)

        assert count > 0
        # Should include message overhead
        assert count > sum(counter.tokenizer.count_tokens(m["content"]) for m in messages)


class TestTokenBudget:
    """Test token budget tracking."""

    def test_add_text_within_budget(self):
        """Test adding text within budget."""
        from rag_factory.utils.token_counter import TokenBudget

        budget = TokenBudget(max_tokens=100, model="gpt-3.5-turbo")

        result = budget.add_text("Short text")

        assert result is True
        assert budget.used > 0
        assert budget.remaining() < 100

    def test_add_text_exceeds_budget(self):
        """Test adding text that exceeds budget."""
        from rag_factory.utils.token_counter import TokenBudget

        budget = TokenBudget(max_tokens=10, model="gpt-3.5-turbo")

        # Add text that's too long
        long_text = "This is a very long text that will definitely exceed the token budget."
        result = budget.add_text(long_text)

        assert result is False
        assert budget.used == 0  # Should not have added

    def test_add_texts(self):
        """Test adding multiple texts."""
        from rag_factory.utils.token_counter import TokenBudget

        budget = TokenBudget(max_tokens=50, model="gpt-3.5-turbo")

        texts = ["Text 1", "Text 2", "Text 3", "Very long text that won't fit" * 10]
        added = budget.add_texts(texts)

        assert len(added) < len(texts)  # Some should be excluded
        assert len(added) >= 3  # First 3 should fit

    def test_reset(self):
        """Test budget reset."""
        from rag_factory.utils.token_counter import TokenBudget

        budget = TokenBudget(max_tokens=100)
        budget.add_text("Some text")

        assert budget.used > 0

        budget.reset()

        assert budget.used == 0
        assert budget.remaining() == budget.available
```

### Integration Tests
```python
# tests/integration/test_tokenization_integration.py
import pytest
from rag_factory.utils.tokenization import Tokenizer, count_tokens


@pytest.mark.integration
class TestTokenizationIntegration:
    """Integration tests for tokenization."""

    def test_openai_token_count_accuracy(self):
        """Test that token counts match OpenAI's counts."""
        # Known token counts from OpenAI
        test_cases = [
            ("Hello, world!", 4),  # GPT-3.5-turbo
            ("The quick brown fox jumps over the lazy dog.", 10),
        ]

        tokenizer = Tokenizer(model_name="gpt-3.5-turbo")

        for text, expected_count in test_cases:
            actual_count = tokenizer.count_tokens(text)
            # Allow small variance
            assert abs(actual_count - expected_count) <= 1

    def test_different_encodings(self):
        """Test different tiktoken encodings."""
        text = "This is a test sentence."

        encodings = ["cl100k_base", "p50k_base", "r50k_base"]

        counts = {}
        for encoding in encodings:
            tokenizer = Tokenizer(encoding_name=encoding)
            counts[encoding] = tokenizer.count_tokens(text)

        # Different encodings should give different counts
        assert len(set(counts.values())) > 1

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
        ]

        tokenizer = Tokenizer()

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text

    def test_multilingual_text(self):
        """Test handling of multilingual text."""
        texts = [
            "Hello",  # English
            "Bonjour",  # French
            "こんにちは",  # Japanese
            "你好",  # Chinese
            "مرحبا",  # Arabic
        ]

        tokenizer = Tokenizer()

        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
```

---

## Documentation

### Tokenization Guide
```markdown
# Tokenization with Tiktoken

## Overview

RAG Factory uses `tiktoken` for fast, lightweight tokenization without requiring the heavy `transformers` library.

## Quick Start

```python
from rag_factory.utils.tokenization import count_tokens, truncate_text

# Count tokens
text = "This is a test document."
token_count = count_tokens(text, model="gpt-3.5-turbo")
print(f"Tokens: {token_count}")

# Truncate to token limit
truncated = truncate_text(text, max_tokens=10, model="gpt-4")
```

## Supported Models

### OpenAI Models
- `gpt-4`, `gpt-4-turbo`: cl100k_base encoding
- `gpt-3.5-turbo`: cl100k_base encoding
- `gpt-3`: p50k_base encoding
- `davinci`, `curie`, `babbage`, `ada`: r50k_base encoding

### Custom Encodings
```python
from rag_factory.utils.tokenization import Tokenizer

# Use specific encoding
tokenizer = Tokenizer(encoding_name="cl100k_base")
```

## Token Budgets

Track token usage against a budget:

```python
from rag_factory.utils.token_counter import TokenBudget

# Create budget
budget = TokenBudget(max_tokens=4000, reserve=500)

# Add texts until budget exhausted
texts = ["Text 1", "Text 2", "Text 3"]
added = budget.add_texts(texts)

print(f"Added {len(added)} texts, {budget.remaining()} tokens remaining")
```

## Text Splitting

Split text by token count:

```python
from rag_factory.utils.tokenization import split_text_by_tokens

chunks = split_text_by_tokens(
    text=long_document,
    chunk_size=512,
    overlap=50,
    model="gpt-3.5-turbo"
)
```

## Fallback Behavior

If tiktoken is unavailable, the system falls back to basic word-based tokenization:

```python
tokenizer = Tokenizer(use_fallback=True)
# Will use fallback if tiktoken fails
```

**Note:** Fallback tokenization is approximate and should only be used for development/testing.
```

---

## Implementation Plan

### Phase 1: Core Tokenization (Days 1-2)
1. Create tokenization utility module
2. Implement Tokenizer class with tiktoken
3. Add encoding caching
4. Implement fallback tokenization
5. Add convenience functions

### Phase 2: Token Utilities (Day 2)
1. Create TokenCounter class
2. Implement TokenBudget class
3. Add message token counting
4. Add text splitting by tokens

### Phase 3: Transformers Removal (Day 3)
1. Find all transformers tokenizer usage
2. Replace with tiktoken
3. Remove transformers from requirements.txt
4. Test all affected code

### Phase 4: Strategy Updates (Day 4)
1. Update Late Chunking strategy
2. Update Contextual RAG strategy
3. Update any other strategies
4. Validate token counting accuracy

### Phase 5: Testing and Documentation (Day 5)
1. Write unit tests
2. Write integration tests
3. Create performance benchmarks
4. Write documentation
5. Create migration guide

---

## Risks and Mitigation

### Risk: Token Count Accuracy
**Impact:** High
**Probability:** Low
**Mitigation:**
- Validate against OpenAI's counts
- Test with known examples
- Document any discrepancies

### Risk: Encoding Compatibility
**Impact:** Medium
**Probability:** Low
**Mitigation:**
- Support multiple encodings
- Provide clear encoding selection
- Test with various models

### Risk: Fallback Quality
**Impact:** Medium
**Probability:** Medium
**Mitigation:**
- Make fallback "good enough" for basic use
- Warn users when using fallback
- Document limitations

---

## Success Metrics

- [ ] Library size reduced by ~500MB (transformers removed)
- [ ] Tokenization speed <10ms for typical documents
- [ ] Token counts match OpenAI's (for OpenAI models)
- [ ] All strategies working with tiktoken
- [ ] All tests passing without transformers
- [ ] Documentation complete

---

## Dependencies

**Blocked by:** None
**Blocks:** None (can run in parallel)
**Related:** Story 10.1 (embedding services), Story 10.3 (late chunking)
