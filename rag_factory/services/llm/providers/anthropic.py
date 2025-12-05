"""Anthropic Claude provider implementation."""

from typing import List, Dict, Any, Iterator
import time
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole


class AnthropicProvider(ILLMProvider):
    """Anthropic Claude provider.

    Supports Claude models including Sonnet, Opus, and Haiku.
    """

    MODELS = {
        "claude-sonnet-4.5": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 3.00,
            "cost_per_1m_completion": 15.00,
        },
        "claude-3-opus-20240229": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 15.00,
            "cost_per_1m_completion": 75.00,
        },
        "claude-3-haiku-20240307": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 0.25,
            "cost_per_1m_completion": 1.25,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic provider.

        Args:
            config: Configuration with 'api_key' and optional 'model'

        Raises:
            ValueError: If model is not supported
        """
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-sonnet-4.5")

        if self.model not in self.MODELS:
            raise ValueError(f"Unknown Anthropic model: {self.model}")

        self.client = Anthropic(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using Claude.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content and metadata
        """
        # Convert messages to Anthropic format
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                conversation.append({"role": msg.role.value, "content": msg.content})

        # Make API call
        start = time.time()

        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        if system_message:
            api_params["system"] = system_message

        response = self.client.messages.create(**api_params)
        latency = time.time() - start

        # Extract content
        content = response.content[0].text if response.content else ""

        # Calculate cost
        cost = self.calculate_cost(
            response.usage.input_tokens, response.usage.output_tokens
        )

        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cost=cost,
            latency=latency,
            metadata={"stop_reason": response.stop_reason},
        )

    def stream(self, messages: List[Message], **kwargs) -> Iterator[StreamChunk]:
        """Generate streaming completion.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            StreamChunk for each piece of generated content
        """
        # Convert messages
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                conversation.append({"role": msg.role.value, "content": msg.content})

        # Stream API call
        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }

        if system_message:
            api_params["system"] = system_message

        with self.client.messages.stream(**api_params) as stream:
            for text in stream.text_stream:
                yield StreamChunk(content=text, is_final=False, metadata={})

            # Final chunk with metadata
            final_message = stream.get_final_message()
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "stop_reason": final_message.stop_reason,
                    "usage": {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                    },
                },
            )

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages.

        Note: Anthropic doesn't have a public tokenizer,
        so we use approximation (1 token ≈ 4 characters).

        Args:
            messages: List of messages

        Returns:
            Approximate number of tokens
        """
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4

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
