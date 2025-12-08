"""
Token counting and budget tracking utilities.

This module provides utilities for counting tokens in various contexts
(messages, function calls) and tracking token usage against budgets.
"""

from typing import Dict, List, Optional, Any
import logging
import json

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

        Based on OpenAI's token counting for chat:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count
        """
        tokens = 0

        for message in messages:
            # Every message has overhead: <im_start>{role}\n{content}<im_end>\n
            tokens += 4

            for key, value in message.items():
                tokens += self.tokenizer.count_tokens(str(value))

        tokens += 2  # Assistant reply priming

        return tokens

    def count_function_call(
        self,
        function_name: str,
        function_args: Dict[str, Any]
    ) -> int:
        """
        Count tokens in a function call.

        Args:
            function_name: Function name
            function_args: Function arguments

        Returns:
            Token count
        """
        # Function call format
        call_str = json.dumps({
            "name": function_name,
            "arguments": function_args
        })

        return self.tokenizer.count_tokens(call_str)
