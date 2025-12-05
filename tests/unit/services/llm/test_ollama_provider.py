"""Unit tests for Ollama provider."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.providers.ollama import OllamaProvider
from rag_factory.services.llm.base import Message, MessageRole


@pytest.fixture
def ollama_config():
    """Create Ollama provider configuration."""
    return {"model": "llama2", "base_url": "http://localhost:11434"}


def test_provider_initialization(ollama_config):
    """Test provider initializes correctly."""
    provider = OllamaProvider(ollama_config)
    assert provider.model == "llama2"
    assert provider.base_url == "http://localhost:11434"


def test_provider_default_values():
    """Test provider uses default values."""
    provider = OllamaProvider({})
    assert provider.model == "llama2"
    assert provider.base_url == "http://localhost:11434"


@patch("httpx.Client")
def test_complete(mock_client_class, ollama_config):
    """Test completion."""
    # Mock httpx response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Hello! How can I help?",
        "total_duration": 1000000,
        "load_duration": 100000,
    }
    mock_client.post.return_value = mock_response
    mock_client_class.return_value = mock_client

    provider = OllamaProvider(ollama_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = provider.complete(messages)

    assert response.content == "Hello! How can I help?"
    assert response.cost == 0.0  # Local models have no cost
    assert response.provider == "ollama"


@patch("httpx.Client")
def test_complete_with_system_message(mock_client_class, ollama_config):
    """Test completion with system message."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Response",
        "total_duration": 1000000,
        "load_duration": 100000,
    }
    mock_client.post.return_value = mock_response
    mock_client_class.return_value = mock_client

    provider = OllamaProvider(ollama_config)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
    ]
    response = provider.complete(messages)

    # Verify prompt includes system message
    call_args = mock_client.post.call_args
    assert "System:" in call_args[1]["json"]["prompt"]


def test_count_tokens(ollama_config):
    """Test token counting."""
    provider = OllamaProvider(ollama_config)

    messages = [Message(role=MessageRole.USER, content="Hello world")]
    count = provider.count_tokens(messages)

    # Approximation: 1 token ≈ 4 characters
    assert count >= 2


def test_calculate_cost(ollama_config):
    """Test cost calculation."""
    provider = OllamaProvider(ollama_config)

    # Local models have no cost
    cost = provider.calculate_cost(1000, 1000)
    assert cost == 0.0


def test_get_model_name(ollama_config):
    """Test getting model name."""
    provider = OllamaProvider(ollama_config)
    assert provider.get_model_name() == "llama2"


def test_get_max_tokens(ollama_config):
    """Test getting max tokens."""
    provider = OllamaProvider(ollama_config)
    assert provider.get_max_tokens() == 4096


def test_messages_to_prompt(ollama_config):
    """Test message to prompt conversion."""
    provider = OllamaProvider(ollama_config)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there"),
        Message(role=MessageRole.USER, content="How are you?"),
    ]

    prompt = provider._messages_to_prompt(messages)

    assert "System: You are helpful" in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there" in prompt
    assert "User: How are you?" in prompt
    assert prompt.endswith("Assistant:")


@patch("httpx.Client")
def test_stream(mock_client_class, ollama_config):
    """Test streaming completion."""
    mock_client = Mock()
    mock_stream = Mock()

    # Mock stream chunks
    mock_stream.iter_lines.return_value = [
        '{"response": "Hello", "done": false}',
        '{"response": " world", "done": false}',
        '{"response": "", "done": true, "total_duration": 1000000}',
    ]
    mock_stream.raise_for_status = Mock()

    mock_client.stream.return_value.__enter__ = Mock(return_value=mock_stream)
    mock_client.stream.return_value.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    provider = OllamaProvider(ollama_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    chunks = list(provider.stream(messages))

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].is_final is True
