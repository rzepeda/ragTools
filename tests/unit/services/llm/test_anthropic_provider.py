"""Unit tests for Anthropic provider."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.providers.anthropic import AnthropicProvider
from rag_factory.services.llm.base import Message, MessageRole


@pytest.fixture
def anthropic_config():
    """Create Anthropic provider configuration."""
    return {"api_key": "test-key", "model": "claude-sonnet-4.5"}


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


@patch("rag_factory.services.llm.providers.anthropic.Anthropic")
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


@patch("rag_factory.services.llm.providers.anthropic.Anthropic")
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
        Message(role=MessageRole.USER, content="Hello"),
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

    expected = (1000 / 1_000_000 * 3.0) + (1000 / 1_000_000 * 15.0)
    assert cost == pytest.approx(expected)


def test_get_model_name(anthropic_config):
    """Test getting model name."""
    provider = AnthropicProvider(anthropic_config)
    assert provider.get_model_name() == "claude-sonnet-4.5"


def test_get_max_tokens(anthropic_config):
    """Test getting max tokens."""
    provider = AnthropicProvider(anthropic_config)
    assert provider.get_max_tokens() == 200000


@patch("rag_factory.services.llm.providers.anthropic.Anthropic")
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
    assert chunks[3].is_final is True
    assert chunks[3].metadata["usage"]["prompt_tokens"] == 10


def test_different_models():
    """Test different model configurations."""
    models = ["claude-sonnet-4.5", "claude-3-opus-20240229", "claude-3-haiku-20240307"]

    for model in models:
        config = {"api_key": "test-key", "model": model}
        provider = AnthropicProvider(config)
        assert provider.get_model_name() == model
        assert provider.get_max_tokens() > 0
