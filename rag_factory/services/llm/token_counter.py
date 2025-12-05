"""Token counting utilities for LLM providers."""

import tiktoken
from typing import List
from .base import Message


class TokenCounter:
    """Utility for counting tokens across different providers."""

    @staticmethod
    def count_openai_tokens(messages: List[Message], model: str = "gpt-4") -> int:
        """Count tokens for OpenAI models using tiktoken.

        Args:
            messages: List of messages
            model: Model name

        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Default to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")

        # Count tokens for each message
        # OpenAI adds tokens for message formatting
        tokens_per_message = 3  # <|start|>role/name\n{content}<|end|>\n
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.content))
            num_tokens += len(encoding.encode(message.role.value))

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

        return num_tokens

    @staticmethod
    def count_anthropic_tokens(messages: List[Message]) -> int:
        """Count tokens for Anthropic models.

        Note: Anthropic doesn't have a public tokenizer,
        so we use approximation (1 token ≈ 4 characters).

        Args:
            messages: List of messages

        Returns:
            Approximate number of tokens
        """
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4

    @staticmethod
    def count_ollama_tokens(messages: List[Message]) -> int:
        """Count tokens for Ollama models.

        Uses character-based approximation.

        Args:
            messages: List of messages

        Returns:
            Approximate number of tokens
        """
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4
