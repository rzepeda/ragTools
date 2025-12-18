"""Ollama local provider implementation."""

from typing import List, Dict, Any, Iterator, AsyncIterator
import time
import httpx

from ..base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole


class OllamaProvider(ILLMProvider):
    """Ollama local model provider.

    Supports local models via Ollama including llama2, mistral, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider.

        Args:
            config: Configuration with optional 'model' and 'base_url'
        """
        self.model = config.get("model", "llama2")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client = httpx.Client(timeout=120.0)

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using Ollama.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content and metadata
        """
        # Convert messages to Ollama format
        prompt = self._messages_to_prompt(messages)

        # Make API call
        start = time.time()

        api_params = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            },
        }

        response = self.client.post(
            f"{self.base_url}/api/generate", json=api_params
        )
        response.raise_for_status()

        latency = time.time() - start
        result = response.json()

        # Extract content
        content = result.get("response", "")

        # Approximate token count
        prompt_tokens = self.count_tokens(messages)
        completion_tokens = len(content) // 4

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=0.0,  # Local models have no cost
            latency=latency,
            metadata={
                "total_duration": result.get("total_duration"),
                "load_duration": result.get("load_duration"),
            },
        )

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            StreamChunk for each piece of generated content
        """
        # Convert messages to Ollama format
        prompt = self._messages_to_prompt(messages)

        # Stream API call
        api_params = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            },
        }

        with self.client.stream(
            "POST", f"{self.base_url}/api/generate", json=api_params
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    import json

                    chunk = json.loads(line)
                    content = chunk.get("response", "")

                    if content:
                        yield StreamChunk(content=content, is_final=False, metadata={})

                    if chunk.get("done", False):
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            metadata={
                                "total_duration": chunk.get("total_duration"),
                                "load_duration": chunk.get("load_duration"),
                            },
                        )

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages.

        Uses character-based approximation.

        Args:
            messages: List of messages

        Returns:
            Approximate number of tokens
        """
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage.

        Local models have no cost.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in dollars (always 0.0)
        """
        return 0.0

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model name string
        """
        return self.model

    def get_max_tokens(self) -> int:
        """Get maximum context window size.

        Returns:
            Maximum number of tokens (default 4096)
        """
        # Default context window, varies by model
        return 4096

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a single prompt string.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
