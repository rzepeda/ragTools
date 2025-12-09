"""Unit tests for LLM service."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole, LLMResponse


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return LLMServiceConfig(
        provider="anthropic", model="claude-sonnet-4.5", enable_rate_limiting=False
    )


@pytest.fixture
def mock_provider():
    """Create mock provider."""
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
        metadata={},
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
        Message(role=MessageRole.USER, content="Hello"),
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
    mock_provider.stream.return_value = iter(
        [
            StreamChunk(content="Hello", is_final=False, metadata={}),
            StreamChunk(content=" world", is_final=False, metadata={}),
            StreamChunk(content="", is_final=True, metadata={}),
        ]
    )

    messages = [Message(role=MessageRole.USER, content="Hello")]
    chunks = list(service.stream(messages))

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[2].is_final is True


def test_stream_with_callback(mock_config, mock_provider, monkeypatch):
    """Test streaming with callback function."""
    from rag_factory.services.llm.base import StreamChunk

    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    mock_provider.stream.return_value = iter(
        [
            StreamChunk(content="Hello", is_final=False, metadata={}),
            StreamChunk(content=" world", is_final=False, metadata={}),
        ]
    )

    collected = []

    def callback(content):
        collected.append(content)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    list(service.stream(messages, callback=callback))

    assert collected == ["Hello", " world"]


def test_stream_empty_messages_raises_error(mock_config):
    """Test streaming with empty messages raises error."""
    service = LLMService(mock_config)

    with pytest.raises(ValueError, match="messages cannot be empty"):
        list(service.stream([]))


def test_provider_error_handling(mock_config, mock_provider, monkeypatch):
    """Test handling of provider errors."""
    service = LLMService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)
    
    mock_provider.complete.side_effect = Exception("API Error")
    
    messages = [Message(role=MessageRole.USER, content="Hello")]
    
    with pytest.raises(Exception, match="API Error"):
        service.complete(messages)


def test_rate_limiter_interaction(mock_config, mock_provider, monkeypatch):
    """Test interaction with rate limiter."""
    mock_config.enable_rate_limiting = True
    
    with patch("rag_factory.services.llm.service.RateLimiter") as mock_limiter_cls:
        mock_limiter = Mock()
        mock_limiter_cls.return_value = mock_limiter
        
        service = LLMService(mock_config)
        monkeypatch.setattr(service, "provider", mock_provider)
        
        messages = [Message(role=MessageRole.USER, content="Hello")]
        service.complete(messages)
        
        mock_limiter.wait_if_needed.assert_called_once()
