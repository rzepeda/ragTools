# Story 3.2: Implement LLM Service Adapter

**Story ID:** 3.2
**Epic:** Epic 3 - Core Services Layer
**Story Points:** 8
**Priority:** Critical
**Dependencies:** None (can be developed in parallel with Story 3.1)

---

## User Story

**As a** system
**I want** a unified interface for LLM calls
**So that** strategies can use different LLM providers interchangeably

---

## Detailed Requirements

### Functional Requirements

1. **Multi-Provider Support**
   - Support for Anthropic Claude (Sonnet 4.5, Opus, Haiku)
   - Support for OpenAI models (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
   - Support for local models via Ollama (llama2, mistral, etc.)
   - Pluggable architecture for adding new providers
   - Provider-specific configuration and features

2. **Prompt Management**
   - Consistent prompt template system
   - Support for system messages, user messages, and assistant messages
   - Template variable substitution
   - Few-shot example support
   - Prompt token counting before sending

3. **Response Handling**
   - Structured response parsing
   - Streaming support for real-time responses
   - Token usage tracking
   - Response validation
   - Metadata extraction (model, tokens, latency)

4. **Token Counting & Cost Tracking**
   - Accurate token counting per provider
   - Cost calculation based on token usage
   - Usage analytics and reporting
   - Budget alerts and limits
   - Historical usage tracking

5. **Rate Limiting & Retries**
   - Rate limiting per provider
   - Exponential backoff on failures
   - Configurable retry attempts
   - Error handling for API failures
   - Request queuing for rate-limited calls

6. **Streaming Support**
   - Real-time token streaming for long responses
   - Callback mechanism for streaming chunks
   - Stream interruption handling
   - Buffering and backpressure management

### Non-Functional Requirements

1. **Performance**
   - Request overhead <50ms (excluding API latency)
   - Streaming latency <100ms to first token
   - Support 100+ concurrent requests
   - Efficient token counting (cached)

2. **Reliability**
   - Automatic retry on transient failures
   - Fallback provider support (optional)
   - Circuit breaker pattern for failing providers
   - Comprehensive error handling

3. **Scalability**
   - Stateless design for horizontal scaling
   - Connection pooling for API clients
   - Async support for high concurrency

4. **Maintainability**
   - Clear provider adapter interface
   - Extensive logging and monitoring
   - Configuration-driven behavior
   - Well-documented API

5. **Security**
   - Secure API key management
   - No sensitive data in logs
   - Input sanitization for prompts
   - PII detection warnings

---

## Acceptance Criteria

### AC1: Provider Support
- [ ] Anthropic Claude provider implemented (Sonnet, Opus, Haiku)
- [ ] OpenAI provider implemented (GPT-4, GPT-3.5)
- [ ] Ollama local provider implemented
- [ ] Provider selection via configuration
- [ ] New providers can be added via plugin interface

### AC2: Prompt Templates
- [ ] Template system with variable substitution
- [ ] Support for system, user, and assistant messages
- [ ] Few-shot examples support
- [ ] Template validation
- [ ] Default templates for common use cases

### AC3: Response Handling
- [ ] Synchronous response handling
- [ ] Streaming response handling
- [ ] Response parsing and validation
- [ ] Metadata extraction (tokens, cost, latency)
- [ ] Error response handling

### AC4: Token Counting
- [ ] Accurate token counting for all providers
- [ ] Token counting before API calls
- [ ] Token usage tracking per request
- [ ] Cost calculation per request
- [ ] Usage statistics available

### AC5: Cost Tracking
- [ ] Cost calculation per provider
- [ ] Total cost tracking
- [ ] Cost per request logged
- [ ] Budget alerts (optional)
- [ ] Cost reporting API

### AC6: Rate Limiting
- [ ] Rate limiter per provider
- [ ] Configurable rate limits
- [ ] Exponential backoff on errors
- [ ] Retry logic with max attempts
- [ ] Request queuing

### AC7: Streaming
- [ ] Streaming support for all providers
- [ ] Callback mechanism for chunks
- [ ] Stream error handling
- [ ] Stream cancellation support
- [ ] Buffering for incomplete tokens

### AC8: Testing
- [ ] Unit tests for all providers with mocked APIs
- [ ] Integration tests with real providers (skippable)
- [ ] Streaming tests
- [ ] Error handling tests
- [ ] Performance benchmarks

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # Base LLM provider interface
│   │   ├── service.py           # Main LLM service
│   │   ├── prompt_template.py   # Prompt template system
│   │   ├── token_counter.py     # Token counting utilities
│   │   ├── rate_limiter.py      # Rate limiting (shared with embedding)
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── anthropic.py     # Anthropic Claude provider
│   │   │   ├── openai.py        # OpenAI provider
│   │   │   └── ollama.py        # Ollama local provider
│   │   └── config.py            # LLM service config
│
tests/
├── unit/
│   └── services/
│       └── llm/
│           ├── test_service.py
│           ├── test_prompt_template.py
│           ├── test_token_counter.py
│           ├── test_anthropic_provider.py
│           ├── test_openai_provider.py
│           └── test_ollama_provider.py
│
├── integration/
│   └── services/
│       └── test_llm_integration.py
```

### Dependencies
```python
# requirements.txt additions
anthropic==0.18.1           # Anthropic Claude API
openai==1.12.0              # OpenAI API
tiktoken==0.5.2             # OpenAI token counting
tenacity==8.2.3             # Retry logic
httpx==0.26.0               # HTTP client for Ollama
```

### Base Provider Interface
```python
# rag_factory/services/llm/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass
from enum import Enum

class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    """Conversation message."""
    role: MessageRole
    content: str

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency: float  # seconds
    metadata: Dict[str, Any]

@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str
    is_final: bool
    metadata: Dict[str, Any]

class ILLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        pass

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion for conversation."""
        pass

    @abstractmethod
    def stream(self, messages: List[Message], **kwargs) -> Iterator[StreamChunk]:
        """Generate streaming completion."""
        pass

    @abstractmethod
    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages."""
        pass

    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum context window size."""
        pass
```

### LLM Service
```python
# rag_factory/services/llm/service.py
from typing import List, Dict, Any, Optional, Iterator, Callable
import time
import logging
from .base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole
from .config import LLMServiceConfig
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class LLMService:
    """
    Centralized LLM service with multi-provider support.

    Example:
        config = LLMServiceConfig(provider="anthropic", model="claude-sonnet-4.5")
        service = LLMService(config)

        messages = [Message(role=MessageRole.USER, content="Hello!")]
        response = service.complete(messages)
        print(response.content)
    """

    def __init__(self, config: LLMServiceConfig):
        self.config = config
        self.provider = self._init_provider()
        self.rate_limiter = RateLimiter(config.rate_limit_config) if config.enable_rate_limiting else None
        self._stats = {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "total_latency": 0.0
        }

    def _init_provider(self) -> ILLMProvider:
        """Initialize LLM provider based on config."""
        from .providers.anthropic import AnthropicProvider
        from .providers.openai import OpenAIProvider
        from .providers.ollama import OllamaProvider

        provider_map = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider
        }

        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return provider_class(self.config.provider_config)

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion for conversation.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        self._stats["total_requests"] += 1

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Count tokens before sending
        prompt_tokens = self.provider.count_tokens(messages)
        logger.info(f"Sending request with {prompt_tokens} prompt tokens")

        # Generate completion
        start = time.time()
        try:
            response = self.provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Update stats
            self._stats["total_prompt_tokens"] += response.prompt_tokens
            self._stats["total_completion_tokens"] += response.completion_tokens
            self._stats["total_cost"] += response.cost
            self._stats["total_latency"] += response.latency

            logger.info(
                f"Completed in {response.latency:.2f}s, "
                f"{response.total_tokens} tokens, "
                f"${response.cost:.6f}"
            )

            return response

        except Exception as e:
            logger.error(f"Error in LLM completion: {e}")
            raise

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """
        Generate streaming completion.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each chunk
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk for each piece of generated content
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        self._stats["total_requests"] += 1

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Stream completion
        try:
            for chunk in self.provider.stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                if callback:
                    callback(chunk.content)

                yield chunk

        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            raise

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages."""
        return self.provider.count_tokens(messages)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_latency = 0.0
        if self._stats["total_requests"] > 0:
            avg_latency = self._stats["total_latency"] / self._stats["total_requests"]

        return {
            **self._stats,
            "average_latency": avg_latency,
            "model": self.provider.get_model_name(),
            "provider": self.config.provider
        }

    def estimate_cost(self, messages: List[Message], max_completion_tokens: int) -> float:
        """Estimate cost for a request."""
        prompt_tokens = self.provider.count_tokens(messages)
        return self.provider.calculate_cost(prompt_tokens, max_completion_tokens)
```

### Anthropic Provider
```python
# rag_factory/services/llm/providers/anthropic.py
from typing import List, Dict, Any, Iterator
import time
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from ..base import ILLMProvider, Message, LLMResponse, StreamChunk, MessageRole

class AnthropicProvider(ILLMProvider):
    """Anthropic Claude provider."""

    MODELS = {
        "claude-sonnet-4.5": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 3.00,
            "cost_per_1m_completion": 15.00
        },
        "claude-3-opus-20240229": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 15.00,
            "cost_per_1m_completion": 75.00
        },
        "claude-3-haiku-20240307": {
            "max_tokens": 200000,
            "cost_per_1m_prompt": 0.25,
            "cost_per_1m_completion": 1.25
        }
    }

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-sonnet-4.5")

        if self.model not in self.MODELS:
            raise ValueError(f"Unknown Anthropic model: {self.model}")

        self.client = Anthropic(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using Claude."""
        # Convert messages to Anthropic format
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        # Make API call
        start = time.time()

        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }

        if system_message:
            api_params["system"] = system_message

        response = self.client.messages.create(**api_params)
        latency = time.time() - start

        # Extract content
        content = response.content[0].text if response.content else ""

        # Calculate cost
        cost = self.calculate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens
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
            metadata={"stop_reason": response.stop_reason}
        )

    def stream(self, messages: List[Message], **kwargs) -> Iterator[StreamChunk]:
        """Generate streaming completion."""
        # Convert messages
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        # Stream API call
        api_params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }

        if system_message:
            api_params["system"] = system_message

        with self.client.messages.stream(**api_params) as stream:
            for text in stream.text_stream:
                yield StreamChunk(
                    content=text,
                    is_final=False,
                    metadata={}
                )

            # Final chunk with metadata
            final_message = stream.get_final_message()
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "stop_reason": final_message.stop_reason,
                    "usage": {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens
                    }
                }
            )

    def count_tokens(self, messages: List[Message]) -> int:
        """
        Count tokens in messages.
        Note: Anthropic doesn't have a public tokenizer,
        so we use approximation (1 token ≈ 4 characters).
        """
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        model_pricing = self.MODELS[self.model]
        prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["cost_per_1m_prompt"]
        completion_cost = (completion_tokens / 1_000_000) * model_pricing["cost_per_1m_completion"]
        return prompt_cost + completion_cost

    def get_model_name(self) -> str:
        return self.model

    def get_max_tokens(self) -> int:
        return self.MODELS[self.model]["max_tokens"]
```

### Prompt Template System
```python
# rag_factory/services/llm/prompt_template.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .base import Message, MessageRole

@dataclass
class PromptTemplate:
    """
    Template for LLM prompts with variable substitution.

    Example:
        template = PromptTemplate(
            system="You are a helpful assistant.",
            user="Answer this question: {question}\nContext: {context}"
        )

        messages = template.format(
            question="What is AI?",
            context="AI stands for Artificial Intelligence..."
        )
    """
    system: Optional[str] = None
    user: Optional[str] = None
    assistant: Optional[str] = None
    few_shot_examples: List[Dict[str, str]] = None

    def format(self, **variables) -> List[Message]:
        """
        Format template with variables.

        Args:
            **variables: Template variables to substitute

        Returns:
            List of formatted messages
        """
        messages = []

        # Add system message
        if self.system:
            content = self._substitute(self.system, variables)
            messages.append(Message(role=MessageRole.SYSTEM, content=content))

        # Add few-shot examples
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append(Message(
                    role=MessageRole.USER,
                    content=self._substitute(example.get("user", ""), variables)
                ))
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=self._substitute(example.get("assistant", ""), variables)
                ))

        # Add user message
        if self.user:
            content = self._substitute(self.user, variables)
            messages.append(Message(role=MessageRole.USER, content=content))

        # Add assistant prefix (if any)
        if self.assistant:
            content = self._substitute(self.assistant, variables)
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))

        return messages

    def _substitute(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in template."""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    def validate(self, **variables) -> bool:
        """Validate that all required variables are provided."""
        try:
            self.format(**variables)
            return True
        except ValueError:
            return False


# Common templates
class CommonTemplates:
    """Collection of common prompt templates."""

    RAG_QA = PromptTemplate(
        system="You are a helpful assistant that answers questions based on provided context.",
        user="""Answer the following question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    )

    SUMMARIZATION = PromptTemplate(
        system="You are an expert at summarizing text concisely.",
        user="""Summarize the following text in {max_words} words or less:

{text}

Summary:"""
    )

    CLASSIFICATION = PromptTemplate(
        system="You are a text classifier.",
        user="""Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
    )

    EXTRACTION = PromptTemplate(
        system="You extract structured information from text.",
        user="""Extract the following information from the text: {fields}

Text: {text}

Extracted information (JSON format):"""
    )
```

---

## Unit Tests

### Test File Locations
- `tests/unit/services/llm/test_service.py`
- `tests/unit/services/llm/test_prompt_template.py`
- `tests/unit/services/llm/test_anthropic_provider.py`

### Test Cases

#### TC3.2.1: LLM Service Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole, LLMResponse

@pytest.fixture
def mock_config():
    return LLMServiceConfig(
        provider="anthropic",
        model="claude-sonnet-4.5",
        enable_rate_limiting=False
    )

@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.complete.return_value = LLMResponse(
        content="Hello! How can I help you?",
        model="claude-sonnet-4.5",
        provider="anthropic",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
        cost=0.0001,
        latency=0.5,
        metadata={}
    )
    provider.count_tokens.return_value = 10
    provider.get_model_name.return_value = "claude-sonnet-4.5"
    return provider

def test_service_initialization(mock_config):
    """Test service initializes correctly."""
    service = LLMService(mock_config)
    assert service.config == mock_config
    assert service.provider is not None

def test_complete_single_message(mock_config, mock_provider, monkeypatch):
    """Test completing with single message."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = service.complete(messages)

    assert response.content == "Hello! How can I help you?"
    assert response.total_tokens == 18
    mock_provider.complete.assert_called_once()

def test_complete_with_system_message(mock_config, mock_provider, monkeypatch):
    """Test completing with system message."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello")
    ]
    response = service.complete(messages)

    assert response is not None
    mock_provider.complete.assert_called_once()

def test_complete_empty_messages_raises_error(mock_config):
    """Test completing with empty messages raises error."""
    service = LLMService(mock_config)

    with pytest.raises(ValueError, match="messages cannot be empty"):
        service.complete([])

def test_complete_with_temperature(mock_config, mock_provider, monkeypatch):
    """Test completing with custom temperature."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    service.complete(messages, temperature=0.5)

    call_args = mock_provider.complete.call_args
    assert call_args[1]["temperature"] == 0.5

def test_complete_with_max_tokens(mock_config, mock_provider, monkeypatch):
    """Test completing with custom max_tokens."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    service.complete(messages, max_tokens=500)

    call_args = mock_provider.complete.call_args
    assert call_args[1]["max_tokens"] == 500

def test_stats_tracking(mock_config, mock_provider, monkeypatch):
    """Test usage statistics tracking."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    service.complete(messages)
    service.complete(messages)

    stats = service.get_stats()

    assert stats["total_requests"] == 2
    assert stats["total_prompt_tokens"] == 20
    assert stats["total_completion_tokens"] == 16
    assert stats["total_cost"] > 0

def test_count_tokens(mock_config, mock_provider, monkeypatch):
    """Test token counting."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    messages = [Message(role=MessageRole.USER, content="Hello world")]
    count = service.count_tokens(messages)

    assert count == 10
    mock_provider.count_tokens.assert_called_once()

def test_estimate_cost(mock_config, mock_provider, monkeypatch):
    """Test cost estimation."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)
    mock_provider.calculate_cost.return_value = 0.002

    messages = [Message(role=MessageRole.USER, content="Hello")]
    cost = service.estimate_cost(messages, max_completion_tokens=1000)

    assert cost == 0.002
    mock_provider.count_tokens.assert_called_once()
    mock_provider.calculate_cost.assert_called_once()

def test_stream_messages(mock_config, mock_provider, monkeypatch):
    """Test streaming completion."""
    from rag_factory.services.llm.base import StreamChunk

    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    # Mock streaming
    mock_provider.stream.return_value = iter([
        StreamChunk(content="Hello", is_final=False, metadata={}),
        StreamChunk(content=" world", is_final=False, metadata={}),
        StreamChunk(content="", is_final=True, metadata={})
    ])

    messages = [Message(role=MessageRole.USER, content="Hello")]
    chunks = list(service.stream(messages))

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[2].is_final == True

def test_stream_with_callback(mock_config, mock_provider, monkeypatch):
    """Test streaming with callback function."""
    from rag_factory.services.llm.base import StreamChunk

    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    mock_provider.stream.return_value = iter([
        StreamChunk(content="Hello", is_final=False, metadata={}),
        StreamChunk(content=" world", is_final=False, metadata={})
    ])

    collected = []
    def callback(content):
        collected.append(content)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    list(service.stream(messages, callback=callback))

    assert collected == ["Hello", " world"]
```

#### TC3.2.2: Prompt Template Tests
```python
import pytest
from rag_factory.services.llm.prompt_template import PromptTemplate, CommonTemplates
from rag_factory.services.llm.base import MessageRole

def test_template_with_system_and_user():
    """Test template with system and user messages."""
    template = PromptTemplate(
        system="You are a {role}",
        user="Hello {name}"
    )

    messages = template.format(role="assistant", name="Alice")

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == "You are a assistant"
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Hello Alice"

def test_template_user_only():
    """Test template with user message only."""
    template = PromptTemplate(user="Question: {question}")

    messages = template.format(question="What is AI?")

    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Question: What is AI?"

def test_template_with_few_shot_examples():
    """Test template with few-shot examples."""
    template = PromptTemplate(
        system="You are a classifier",
        few_shot_examples=[
            {"user": "Text: Happy day", "assistant": "Positive"},
            {"user": "Text: Sad day", "assistant": "Negative"}
        ],
        user="Text: {text}"
    )

    messages = template.format(text="Great weather")

    assert len(messages) == 6  # system + 2 examples (2 msgs each) + user
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[5].content == "Text: Great weather"

def test_template_missing_variable_raises_error():
    """Test template with missing variable raises error."""
    template = PromptTemplate(user="Hello {name}")

    with pytest.raises(ValueError, match="Missing template variable"):
        template.format()

def test_template_validation():
    """Test template validation."""
    template = PromptTemplate(user="Hello {name}")

    assert template.validate(name="Alice") == True
    assert template.validate() == False

def test_common_template_rag_qa():
    """Test RAG QA common template."""
    messages = CommonTemplates.RAG_QA.format(
        context="The sky is blue.",
        question="What color is the sky?"
    )

    assert len(messages) == 2
    assert "context" in messages[1].content.lower()
    assert "question" in messages[1].content.lower()

def test_common_template_summarization():
    """Test summarization common template."""
    messages = CommonTemplates.SUMMARIZATION.format(
        text="Long text here...",
        max_words="50"
    )

    assert len(messages) == 2
    assert "summarize" in messages[1].content.lower()

def test_template_multiple_variables():
    """Test template with multiple variables."""
    template = PromptTemplate(
        user="Name: {name}, Age: {age}, City: {city}"
    )

    messages = template.format(name="Alice", age=30, city="NYC")

    assert messages[0].content == "Name: Alice, Age: 30, City: NYC"
```

#### TC3.2.3: Anthropic Provider Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.providers.anthropic import AnthropicProvider
from rag_factory.services.llm.base import Message, MessageRole

@pytest.fixture
def anthropic_config():
    return {
        "api_key": "test-key",
        "model": "claude-sonnet-4.5"
    }

def test_provider_initialization(anthropic_config):
    """Test provider initializes correctly."""
    provider = AnthropicProvider(anthropic_config)
    assert provider.model == "claude-sonnet-4.5"
    assert provider.get_max_tokens() == 200000

def test_provider_invalid_model():
    """Test provider raises error for invalid model."""
    config = {"api_key": "test-key", "model": "invalid-model"}

    with pytest.raises(ValueError, match="Unknown Anthropic model"):
        AnthropicProvider(config)

@patch("anthropic.Anthropic")
def test_complete(mock_anthropic_class, anthropic_config):
    """Test completion."""
    # Mock Anthropic response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Hello! How can I help?")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 8
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    provider = AnthropicProvider(anthropic_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = provider.complete(messages)

    assert response.content == "Hello! How can I help?"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 8
    assert response.provider == "anthropic"

@patch("anthropic.Anthropic")
def test_complete_with_system_message(mock_anthropic_class, anthropic_config):
    """Test completion with system message."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Response")]
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 5
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    provider = AnthropicProvider(anthropic_config)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello")
    ]
    response = provider.complete(messages)

    # Verify system message was passed separately
    call_args = mock_client.messages.create.call_args
    assert "system" in call_args[1]
    assert call_args[1]["system"] == "You are helpful"

def test_count_tokens(anthropic_config):
    """Test token counting."""
    provider = AnthropicProvider(anthropic_config)

    messages = [Message(role=MessageRole.USER, content="Hello world")]
    count = provider.count_tokens(messages)

    # Approximation: 1 token ≈ 4 characters
    # "Hello world" = 11 chars ≈ 2-3 tokens
    assert count >= 2

def test_calculate_cost(anthropic_config):
    """Test cost calculation."""
    provider = AnthropicProvider(anthropic_config)

    # Claude Sonnet 4.5: $3/1M prompt tokens, $15/1M completion tokens
    cost = provider.calculate_cost(1000, 1000)

    expected = (1000/1_000_000 * 3.0) + (1000/1_000_000 * 15.0)
    assert cost == pytest.approx(expected)

def test_get_model_name(anthropic_config):
    """Test getting model name."""
    provider = AnthropicProvider(anthropic_config)
    assert provider.get_model_name() == "claude-sonnet-4.5"

def test_get_max_tokens(anthropic_config):
    """Test getting max tokens."""
    provider = AnthropicProvider(anthropic_config)
    assert provider.get_max_tokens() == 200000

@patch("anthropic.Anthropic")
def test_stream(mock_anthropic_class, anthropic_config):
    """Test streaming completion."""
    mock_client = Mock()
    mock_stream = Mock()

    # Mock text stream
    mock_stream.text_stream = iter(["Hello", " ", "world"])

    # Mock final message
    mock_final = Mock()
    mock_final.stop_reason = "end_turn"
    mock_final.usage.input_tokens = 10
    mock_final.usage.output_tokens = 3
    mock_stream.get_final_message.return_value = mock_final

    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)

    mock_client.messages.stream.return_value = mock_stream
    mock_anthropic_class.return_value = mock_client

    provider = AnthropicProvider(anthropic_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    chunks = list(provider.stream(messages))

    assert len(chunks) == 4  # 3 content chunks + 1 final
    assert chunks[0].content == "Hello"
    assert chunks[3].is_final == True
    assert chunks[3].metadata["usage"]["prompt_tokens"] == 10
```

---

## Integration Tests

### Test File Location
`tests/integration/services/test_llm_integration.py`

### Test Scenarios

#### IS3.2.1: End-to-End LLM Workflow
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="API key not set")
def test_full_llm_workflow():
    """Test complete LLM workflow with real API."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",  # Use cheaper model for testing
        provider_config={
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        },
        enable_rate_limiting=True
    )

    service = LLMService(config)

    # Test simple completion
    messages = [Message(role=MessageRole.USER, content="Say hello in 3 words.")]
    response = service.complete(messages, max_tokens=20)

    assert response.content
    assert response.total_tokens > 0
    assert response.cost > 0
    assert response.latency > 0

    # Test with system message
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a concise assistant."),
        Message(role=MessageRole.USER, content="What is 2+2?")
    ]
    response2 = service.complete(messages, temperature=0.0)

    assert "4" in response2.content

    # Check stats
    stats = service.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_cost"] > 0

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="API key not set")
def test_streaming_response():
    """Test streaming LLM response."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        }
    )

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Count from 1 to 5.")]

    collected_content = []
    for chunk in service.stream(messages, max_tokens=50):
        if not chunk.is_final:
            collected_content.append(chunk.content)

    full_response = "".join(collected_content)
    assert len(full_response) > 0
    # Should contain numbers
    assert any(str(i) in full_response for i in range(1, 6))

@pytest.mark.integration
def test_local_ollama_provider():
    """Test local Ollama provider."""
    # Skip if Ollama not running
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        if response.status_code != 200:
            pytest.skip("Ollama not running")
    except:
        pytest.skip("Ollama not available")

    config = LLMServiceConfig(
        provider="ollama",
        model="llama2",
        enable_rate_limiting=False
    )

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Say hello.")]
    response = service.complete(messages, max_tokens=20)

    assert response.content
    assert response.cost == 0.0  # Local models have no cost
    assert response.provider == "ollama"

@pytest.mark.integration
def test_prompt_template_with_real_llm():
    """Test prompt template with real LLM."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API key not set")

    from rag_factory.services.llm.prompt_template import CommonTemplates

    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")}
    )

    service = LLMService(config)

    # Use RAG QA template
    messages = CommonTemplates.RAG_QA.format(
        context="Paris is the capital of France. It is known for the Eiffel Tower.",
        question="What is Paris known for?"
    )

    response = service.complete(messages, max_tokens=100)

    assert "Eiffel Tower" in response.content or "eiffel tower" in response.content.lower()

@pytest.mark.integration
def test_concurrent_llm_requests():
    """Test concurrent LLM requests."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API key not set")

    import concurrent.futures

    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
        enable_rate_limiting=True,
        rate_limit_config={"requests_per_minute": 50}
    )

    service = LLMService(config)

    def make_request(i):
        messages = [Message(role=MessageRole.USER, content=f"Say number {i}")]
        return service.complete(messages, max_tokens=10)

    # Run 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 10
    assert all(r.content for r in results)
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_llm_performance.py

import pytest
import time
from rag_factory.services.llm import LLMService, LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole

@pytest.mark.benchmark
def test_request_overhead():
    """Test request overhead is <50ms."""
    # Use mock provider to isolate overhead
    from unittest.mock import Mock

    config = LLMServiceConfig(provider="anthropic", enable_rate_limiting=False)
    service = LLMService(config)

    # Mock provider
    mock_response = Mock()
    mock_response.content = "test"
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.total_tokens = 15
    mock_response.cost = 0.0
    mock_response.latency = 0.0
    mock_response.metadata = {}

    service.provider.complete = Mock(return_value=mock_response)
    service.provider.count_tokens = Mock(return_value=10)

    messages = [Message(role=MessageRole.USER, content="test")]

    # Warm up
    service.complete(messages)

    # Benchmark
    start = time.time()
    service.complete(messages)
    overhead = (time.time() - start) * 1000

    assert overhead < 50, f"Overhead {overhead:.2f}ms (expected <50ms)"

@pytest.mark.benchmark
def test_concurrent_request_handling():
    """Test handling 100+ concurrent requests."""
    import concurrent.futures
    from unittest.mock import Mock

    config = LLMServiceConfig(provider="anthropic", enable_rate_limiting=False)
    service = LLMService(config)

    # Mock provider
    mock_response = Mock()
    mock_response.content = "test"
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.total_tokens = 15
    mock_response.cost = 0.0
    mock_response.latency = 0.0
    mock_response.metadata = {}

    service.provider.complete = Mock(return_value=mock_response)
    service.provider.count_tokens = Mock(return_value=10)

    def make_request(i):
        messages = [Message(role=MessageRole.USER, content=f"test {i}")]
        return service.complete(messages)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_request, i) for i in range(100)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    duration = time.time() - start

    assert len(results) == 100
    assert duration < 5.0, f"100 requests took {duration:.2f}s (expected <5s)"
```

---

## Definition of Done

- [ ] Base LLM provider interface defined
- [ ] Anthropic Claude provider fully implemented
- [ ] OpenAI provider fully implemented
- [ ] Ollama local provider implemented
- [ ] LLM service with rate limiting implemented
- [ ] Prompt template system implemented
- [ ] Streaming support working
- [ ] Token counting implemented
- [ ] Cost tracking implemented
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Configuration system working
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install anthropic openai tiktoken tenacity httpx

# For Ollama (local models)
# Download and install from https://ollama.ai
# Then pull a model:
ollama pull llama2
```

### Configuration

```python
# config.yaml
llm_service:
  provider: "anthropic"  # anthropic, openai, or ollama
  model: "claude-sonnet-4.5"

  provider_config:
    api_key: "${ANTHROPIC_API_KEY}"

  rate_limiting:
    enabled: true
    requests_per_minute: 50
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### Usage Example

```python
from rag_factory.services.llm import LLMService, LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole
from rag_factory.services.llm.prompt_template import CommonTemplates

# Initialize service
config = LLMServiceConfig(
    provider="anthropic",
    model="claude-sonnet-4.5"
)
service = LLMService(config)

# Simple completion
messages = [Message(role=MessageRole.USER, content="What is AI?")]
response = service.complete(messages, max_tokens=200)
print(response.content)

# Using templates
messages = CommonTemplates.RAG_QA.format(
    context="AI is artificial intelligence...",
    question="What is AI?"
)
response = service.complete(messages)
print(response.content)

# Streaming
for chunk in service.stream(messages):
    if not chunk.is_final:
        print(chunk.content, end="", flush=True)

# Check stats
stats = service.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
```

---

## Notes for Developers

1. **Provider Selection**: Use Haiku for development to minimize costs, Sonnet for production.

2. **Token Counting**: Always count tokens before sending to avoid exceeding limits.

3. **Cost Management**: Monitor costs using built-in tracking. Set budget alerts.

4. **Rate Limiting**: Configure rate limits based on your API tier.

5. **Streaming**: Use streaming for long responses to provide better UX.

6. **Error Handling**: The service includes automatic retries with exponential backoff.

7. **Testing**: Use mocked providers for unit tests to avoid API costs.

8. **Prompt Templates**: Use templates for consistent prompting across strategies.

9. **Model Selection**: Choose model based on requirements:
   - Haiku: Fast, cheap, good for simple tasks
   - Sonnet: Balanced performance and cost
   - Opus: Best quality, most expensive

10. **Local Models**: Use Ollama for development and testing without API costs.
