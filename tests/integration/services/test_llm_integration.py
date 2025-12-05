"""Integration tests for LLM service."""

import os
import pytest
from rag_factory.services.llm import LLMService, LLMServiceConfig, Message, MessageRole
from rag_factory.services.llm.prompt_template import CommonTemplates


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_full_llm_workflow():
    """Test complete LLM workflow with real API."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",  # Use cheaper model for testing
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
        enable_rate_limiting=True,
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
        Message(role=MessageRole.USER, content="What is 2+2?"),
    ]
    response2 = service.complete(messages, temperature=0.0)

    assert "4" in response2.content

    # Check stats
    stats = service.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_cost"] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_streaming_response():
    """Test streaming LLM response."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
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
    except Exception:
        pytest.skip("Ollama not available")

    config = LLMServiceConfig(provider="ollama", model="llama2", enable_rate_limiting=False)

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Say hello.")]
    response = service.complete(messages, max_tokens=20)

    assert response.content
    assert response.cost == 0.0  # Local models have no cost
    assert response.provider == "ollama"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_prompt_template_with_real_llm():
    """Test prompt template with real LLM."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
    )

    service = LLMService(config)

    # Use RAG QA template
    messages = CommonTemplates.RAG_QA.format(
        context="Paris is the capital of France. It is known for the Eiffel Tower.",
        question="What is Paris known for?",
    )

    response = service.complete(messages, max_tokens=100)

    assert "Eiffel Tower" in response.content or "eiffel tower" in response.content.lower()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_provider():
    """Test OpenAI provider integration."""
    config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")},
        enable_rate_limiting=True,
    )

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Say hello in 3 words.")]
    response = service.complete(messages, max_tokens=20)

    assert response.content
    assert response.total_tokens > 0
    assert response.cost > 0
    assert response.provider == "openai"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_concurrent_llm_requests():
    """Test concurrent LLM requests."""
    import concurrent.futures

    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
        enable_rate_limiting=True,
        rate_limit_config={"requests_per_minute": 50},
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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_cost_tracking():
    """Test cost tracking functionality."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
    )

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Hello")]

    # Estimate cost before making request
    estimated_cost = service.estimate_cost(messages, max_completion_tokens=100)
    assert estimated_cost > 0

    # Make actual request
    response = service.complete(messages, max_tokens=100)

    # Verify cost tracking
    stats = service.get_stats()
    assert stats["total_cost"] > 0
    assert stats["total_requests"] == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_token_counting():
    """Test token counting accuracy."""
    config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")},
    )

    service = LLMService(config)

    messages = [Message(role=MessageRole.USER, content="Hello world")]

    # Count tokens
    token_count = service.count_tokens(messages)
    assert token_count > 0

    # Make request and compare
    response = service.complete(messages, max_tokens=10)
    # Token counts should be reasonably close (within approximation)
    assert abs(response.prompt_tokens - token_count) < 10
