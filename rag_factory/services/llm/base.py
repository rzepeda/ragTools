"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, AsyncIterator
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Conversation message.

    Attributes:
        role: Role of the message sender
        content: Text content of the message
    """

    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    """Response from LLM.

    Attributes:
        content: Generated text content
        model: Name of the model used
        provider: Name of the provider
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens
        cost: Cost in dollars for this request
        latency: Request latency in seconds
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency: float
    metadata: Dict[str, Any]


@dataclass
class StreamChunk:
    """Streaming response chunk.

    Attributes:
        content: Chunk of generated text
        is_final: Whether this is the final chunk
        metadata: Additional chunk metadata
    """

    content: str
    is_final: bool
    metadata: Dict[str, Any]


class ILLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure
    consistent behavior across different providers.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion for conversation.

        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            Exception: If completion generation fails
        """

    @abstractmethod
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion.

        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk for each piece of generated content

        Raises:
            Exception: If streaming fails
        """

    @abstractmethod
    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """

    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in dollars
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Name of the model being used
        """

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum context window size.

        Returns:
            Maximum number of tokens in context window
        """
