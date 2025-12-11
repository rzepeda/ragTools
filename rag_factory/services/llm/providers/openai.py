"""OpenAI provider implementation."""

from typing import List, Dict, Any, Iterator
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole
from ..token_counter import TokenCounter


class OpenAIProvider(ILLMProvider):
    """OpenAI provider.

    Supports GPT-4, GPT-4-turbo, and GPT-3.5-turbo models.
    """

    MODELS = {
        "gpt-4": {
            "max_tokens": 8192,
            "cost_per_1m_prompt": 30.00,
            "cost_per_1m_completion": 60.00,
        },
        "gpt-4-turbo": {
            "max_tokens": 128000,
            "cost_per_1m_prompt": 10.00,
            "cost_per_1m_completion": 30.00,
        },
        "gpt-4-turbo-preview": {
            "max_tokens": 128000,
            "cost_per_1m_prompt": 10.00,
            "cost_per_1m_completion": 30.00,
        },
        "gpt-3.5-turbo": {
            "max_tokens": 16385,
            "cost_per_1m_prompt": 0.50,
            "cost_per_1m_completion": 1.50,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider.

        Args:
            config: Configuration with 'api_key', optional 'model' and 'base_url'

        Raises:
            ValueError: If model is not supported
        """
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4-turbo")
        self.base_url = config.get("base_url")  # For LM Studio compatibility

        # Only validate model if using OpenAI API (not LM Studio)
        if not self.base_url and self.model not in self.MODELS:
            raise ValueError(f"Unknown OpenAI model: {self.model}")

        # Use custom base_url if provided (for LM Studio)
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using OpenAI.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content and metadata
        """
        # Convert messages to OpenAI format
        conversation = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        # Make API call
        start = time.time()

        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        response = self.client.chat.completions.create(**api_params)
        latency = time.time() - start

        # Extract content
        content = response.choices[0].message.content or ""

        # Calculate cost
        cost = self.calculate_cost(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )

        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=cost,
            latency=latency,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )

    def stream(self, messages: List[Message], **kwargs) -> Iterator[StreamChunk]:
        """Generate streaming completion.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            StreamChunk for each piece of generated content
        """
        # Convert messages to OpenAI format
        conversation = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        # Stream API call
        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }

        stream = self.client.chat.completions.create(**api_params)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    is_final=False,
                    metadata={},
                )

            # Check if this is the final chunk
            if chunk.choices[0].finish_reason:
                yield StreamChunk(
                    content="",
                    is_final=True,
                    metadata={"finish_reason": chunk.choices[0].finish_reason},
                )

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages using tiktoken.

        Args:
            messages: List of messages

        Returns:
            Number of tokens
        """
        return TokenCounter.count_openai_tokens(messages, self.model)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in dollars
        """
        model_pricing = self.MODELS[self.model]
        prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["cost_per_1m_prompt"]
        completion_cost = (
            completion_tokens / 1_000_000
        ) * model_pricing["cost_per_1m_completion"]
        return prompt_cost + completion_cost

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model name string
        """
        return self.model

    def get_max_tokens(self) -> int:
        """Get maximum context window size.

        Returns:
            Maximum number of tokens
        """
        return self.MODELS[self.model]["max_tokens"]
