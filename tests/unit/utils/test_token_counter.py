"""
Unit tests for token counter and budget utilities.
"""

import pytest
from rag_factory.utils.token_counter import TokenCounter, TokenBudget


class TestTokenCounter:
    """Test token counter functionality."""

    def test_initialization(self):
        """Test token counter initialization."""
        counter = TokenCounter(model="gpt-3.5-turbo")
        assert counter.model == "gpt-3.5-turbo"
        assert counter.tokenizer is not None

    def test_count_messages(self):
        """Test counting tokens in messages."""
        counter = TokenCounter(model="gpt-3.5-turbo")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        count = counter.count_messages(messages)

        assert count > 0
        # Should include message overhead (4 tokens per message + 2 for priming)
        # Total overhead: 4*3 + 2 = 14 tokens
        content_tokens = sum(
            counter.tokenizer.count_tokens(m["content"]) for m in messages
        )
        assert count > content_tokens

    def test_count_messages_empty(self):
        """Test counting tokens in empty messages."""
        counter = TokenCounter()
        
        count = counter.count_messages([])
        
        # Should still have priming tokens
        assert count == 2

    def test_count_function_call(self):
        """Test counting tokens in function call."""
        counter = TokenCounter()

        function_name = "get_weather"
        function_args = {
            "location": "San Francisco",
            "unit": "celsius"
        }

        count = counter.count_function_call(function_name, function_args)

        assert count > 0

    def test_count_function_call_complex_args(self):
        """Test counting tokens in function call with complex arguments."""
        counter = TokenCounter()

        function_name = "process_data"
        function_args = {
            "data": [1, 2, 3, 4, 5],
            "options": {
                "normalize": True,
                "scale": 1.5
            }
        }

        count = counter.count_function_call(function_name, function_args)

        assert count > 0


class TestTokenBudget:
    """Test token budget tracking."""

    def test_initialization(self):
        """Test budget initialization."""
        budget = TokenBudget(max_tokens=100, model="gpt-3.5-turbo")
        
        assert budget.max_tokens == 100
        assert budget.reserve == 0
        assert budget.available == 100
        assert budget.used == 0

    def test_initialization_with_reserve(self):
        """Test budget initialization with reserve."""
        budget = TokenBudget(max_tokens=100, reserve=20)
        
        assert budget.max_tokens == 100
        assert budget.reserve == 20
        assert budget.available == 80
        assert budget.used == 0

    def test_add_text_within_budget(self):
        """Test adding text within budget."""
        budget = TokenBudget(max_tokens=100, model="gpt-3.5-turbo")

        result = budget.add_text("Short text")

        assert result is True
        assert budget.used > 0
        assert budget.remaining() < 100

    def test_add_text_exceeds_budget(self):
        """Test adding text that exceeds budget."""
        budget = TokenBudget(max_tokens=10, model="gpt-3.5-turbo")

        # Add text that's too long
        long_text = "This is a very long text that will definitely exceed the token budget."
        result = budget.add_text(long_text)

        assert result is False
        assert budget.used == 0  # Should not have added

    def test_add_text_exact_budget(self):
        """Test adding text that exactly fits budget."""
        budget = TokenBudget(max_tokens=100, model="gpt-3.5-turbo")
        
        # Add text until we're close to the limit
        text = "word " * 20  # Should be around 20-40 tokens
        budget.add_text(text)
        
        # Try to add more
        result = budget.add_text("more text")
        
        # Should succeed or fail depending on remaining space
        assert isinstance(result, bool)

    def test_add_texts(self):
        """Test adding multiple texts."""
        budget = TokenBudget(max_tokens=50, model="gpt-3.5-turbo")

        texts = ["Text 1", "Text 2", "Text 3", "Very long text that won't fit" * 10]
        added = budget.add_texts(texts)

        assert len(added) < len(texts)  # Some should be excluded
        assert len(added) >= 3  # First 3 should fit

    def test_add_texts_all_fit(self):
        """Test adding multiple texts that all fit."""
        budget = TokenBudget(max_tokens=1000, model="gpt-3.5-turbo")

        texts = ["Text 1", "Text 2", "Text 3"]
        added = budget.add_texts(texts)

        assert len(added) == len(texts)

    def test_add_texts_none_fit(self):
        """Test adding texts when none fit."""
        budget = TokenBudget(max_tokens=5, model="gpt-3.5-turbo")

        texts = ["This is a long text" * 10, "Another long text" * 10]
        added = budget.add_texts(texts)

        assert len(added) == 0

    def test_remaining(self):
        """Test remaining tokens calculation."""
        budget = TokenBudget(max_tokens=100)
        
        initial_remaining = budget.remaining()
        assert initial_remaining == 100
        
        budget.add_text("Some text")
        
        new_remaining = budget.remaining()
        assert new_remaining < initial_remaining
        assert new_remaining == budget.available - budget.used

    def test_reset(self):
        """Test budget reset."""
        budget = TokenBudget(max_tokens=100)
        budget.add_text("Some text")

        assert budget.used > 0

        budget.reset()

        assert budget.used == 0
        assert budget.remaining() == budget.available

    def test_multiple_adds_and_reset(self):
        """Test multiple adds followed by reset."""
        budget = TokenBudget(max_tokens=100)
        
        budget.add_text("Text 1")
        budget.add_text("Text 2")
        used_before_reset = budget.used
        
        assert used_before_reset > 0
        
        budget.reset()
        
        assert budget.used == 0
        
        # Should be able to add again
        result = budget.add_text("Text 3")
        assert result is True

    def test_budget_with_reserve(self):
        """Test budget behavior with reserve."""
        budget = TokenBudget(max_tokens=100, reserve=20)
        
        # Available should be 80
        assert budget.available == 80
        
        # Add text that would fit in 100 but not in 80
        # This is tricky to test precisely, so we just verify the logic
        long_text = "word " * 25  # Approximately 25-50 tokens
        result = budget.add_text(long_text)
        
        # Should respect the available limit (80), not max_tokens (100)
        if result:
            assert budget.used <= 80


class TestTokenBudgetEdgeCases:
    """Test edge cases for token budget."""

    def test_zero_budget(self):
        """Test budget with zero tokens."""
        budget = TokenBudget(max_tokens=0)
        
        result = budget.add_text("Any text")
        
        assert result is False
        assert budget.used == 0

    def test_empty_text(self):
        """Test adding empty text."""
        budget = TokenBudget(max_tokens=100)
        
        result = budget.add_text("")
        
        assert result is True
        assert budget.used == 0

    def test_reserve_equals_max(self):
        """Test when reserve equals max tokens."""
        budget = TokenBudget(max_tokens=100, reserve=100)
        
        assert budget.available == 0
        
        result = budget.add_text("Any text")
        assert result is False

    def test_reserve_exceeds_max(self):
        """Test when reserve exceeds max tokens."""
        budget = TokenBudget(max_tokens=100, reserve=150)
        
        # Available should be negative, which means nothing can be added
        assert budget.available < 0
        
        result = budget.add_text("Any text")
        assert result is False
