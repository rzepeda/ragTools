"""Unit tests for OpenAI provider."""

import pytest
from unittest.mock import Mock, patch
from rag_factory.services.llm.providers.openai import OpenAIProvider
from rag_factory.services.llm.base import Message, MessageRole


@pytest.fixture
def openai_config():
    """Create OpenAI provider configuration."""
    return {"api_key": "test-key", "model": "gpt-4-turbo"}


def test_provider_initialization(openai_config):
    """Test provider initializes correctly."""
    provider = OpenAIProvider(openai_config)
    assert provider.model == "gpt-4-turbo"
    assert provider.get_max_tokens() == 128000


def test_provider_invalid_model():
    """Test provider raises error for invalid model."""
    config = {"api_key": "test-key", "model": "invalid-model"}

    with pytest.raises(ValueError, match="Unknown OpenAI model"):
        OpenAIProvider(config)


@patch("rag_factory.services.llm.providers.openai.OpenAI")
def test_complete(mock_openai_class, openai_config):
    """Test completion."""
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Hello! How can I help?"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 18
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    provider = OpenAIProvider(openai_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = provider.complete(messages)

    assert response.content == "Hello! How can I help?"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 8
    assert response.provider == "openai"


@patch("rag_factory.services.llm.providers.openai.OpenAI")
def test_complete_with_system_message(mock_openai_class, openai_config):
    """Test completion with system message."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 20
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    provider = OpenAIProvider(openai_config)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
    ]
    response = provider.complete(messages)

    # Verify messages were passed correctly
    call_args = mock_client.chat.completions.create.call_args
    assert len(call_args[1]["messages"]) == 2
    assert call_args[1]["messages"][0]["role"] == "system"


@patch("rag_factory.services.llm.token_counter.TokenCounter.count_openai_tokens")
def test_count_tokens(mock_count, openai_config):
    """Test token counting."""
    mock_count.return_value = 10

    provider = OpenAIProvider(openai_config)
    messages = [Message(role=MessageRole.USER, content="Hello world")]
    count = provider.count_tokens(messages)

    assert count == 10
    mock_count.assert_called_once()


def test_calculate_cost(openai_config):
    """Test cost calculation."""
    provider = OpenAIProvider(openai_config)

    # GPT-4-turbo: $10/1M prompt tokens, $30/1M completion tokens
    cost = provider.calculate_cost(1000, 1000)

    expected = (1000 / 1_000_000 * 10.0) + (1000 / 1_000_000 * 30.0)
    assert cost == pytest.approx(expected)


def test_get_model_name(openai_config):
    """Test getting model name."""
    provider = OpenAIProvider(openai_config)
    assert provider.get_model_name() == "gpt-4-turbo"


def test_get_max_tokens(openai_config):
    """Test getting max tokens."""
    provider = OpenAIProvider(openai_config)
    assert provider.get_max_tokens() == 128000


@pytest.mark.asyncio
@patch("rag_factory.services.llm.providers.openai.OpenAI")
async def test_stream(mock_openai_class, openai_config):
    """Test streaming completion."""
    mock_client = Mock()

    # Mock stream chunks
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.choices[0].finish_reason = None

    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.choices[0].finish_reason = None

    chunk3 = Mock()
    chunk3.choices = [Mock()]
    chunk3.choices[0].delta.content = None
    chunk3.choices[0].finish_reason = "stop"

    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
    mock_openai_class.return_value = mock_client

    provider = OpenAIProvider(openai_config)

    messages = [Message(role=MessageRole.USER, content="Hello")]
    chunks = []
    async for chunk in provider.stream(messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].is_final is True


def test_different_models():
    """Test different model configurations."""
    models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

    for model in models:
        config = {"api_key": "test-key", "model": model}
        provider = OpenAIProvider(config)
        assert provider.get_model_name() == model
        assert provider.get_max_tokens() > 0
