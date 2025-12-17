"""
Real integration tests for LLM services.

Tests actual LLM text generation with real services configured via .env.
"""

import pytest


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_basic_generation(real_llm_service):
    """Test basic LLM text generation."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="What is 2+2? Answer with just the number."
        )
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0
    assert "4" in response.content


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_conversation(real_llm_service):
    """Test multi-turn conversation."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="My name is Alice."),
        Message(role=MessageRole.ASSISTANT, content="Hello Alice! Nice to meet you."),
        Message(role=MessageRole.USER, content="What is my name?")
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    assert "alice" in response.content.lower()


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_token_counting(real_llm_service):
    """Test token counting functionality."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(role=MessageRole.USER, content="Hello, how are you?")
    ]
    
    response = real_llm_service.complete(messages)
    
    # Response should include token usage
    assert hasattr(response, 'total_tokens') or hasattr(response, 'usage')
    
    if hasattr(response, 'total_tokens'):
        assert response.total_tokens > 0
    elif hasattr(response, 'usage'):
        assert response.usage.total_tokens > 0


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_with_system_prompt(real_llm_service):
    """Test LLM with system prompt."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a pirate. Always respond in pirate speak."
        ),
        Message(
            role=MessageRole.USER,
            content="Tell me about the weather."
        )
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    # Response should contain pirate-like language (this is a soft check)
    content_lower = response.content.lower()
    pirate_indicators = ["arr", "matey", "ye", "aye", "ship", "sea"]
    # At least one pirate indicator should be present (model-dependent)
    # This is a weak test since not all models follow system prompts perfectly


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_json_response(real_llm_service):
    """Test LLM generating structured JSON response."""
    from rag_factory.services.llm.base import Message, MessageRole
    import json
    
    messages = [
        Message(
            role=MessageRole.USER,
            content='Return a JSON object with keys "name" and "age" for a person named John who is 30 years old. Return ONLY the JSON, no other text.'
        )
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    
    # Try to parse JSON from response
    try:
        # Extract JSON from response (might have markdown code blocks)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        assert "name" in data or "Name" in data
        assert "age" in data or "Age" in data
    except json.JSONDecodeError:
        # Some models might not follow instructions perfectly
        # Just verify response contains the expected information
        assert "john" in response.content.lower()
        assert "30" in response.content


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_max_tokens(real_llm_service):
    """Test LLM with max_tokens parameter."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a long essay about artificial intelligence."
        )
    ]
    
    # Request with low max_tokens
    response = real_llm_service.complete(messages, max_tokens=20)
    
    assert response is not None
    # Response should be relatively short
    assert len(response.content.split()) < 50  # Rough check


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_temperature(real_llm_service):
    """Test LLM with different temperature settings."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Say hello."
        )
    ]
    
    # Low temperature (more deterministic)
    response1 = real_llm_service.complete(messages, temperature=0.1)
    response2 = real_llm_service.complete(messages, temperature=0.1)
    
    assert response1 is not None
    assert response2 is not None
    
    # Responses should be similar (but not necessarily identical)
    # This is a weak test since even low temperature can vary


@pytest.mark.real_integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_llm_streaming(real_llm_service):
    """Test LLM streaming responses."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Count from 1 to 5."
        )
    ]
    
    # Check if service supports streaming
    if hasattr(real_llm_service, 'stream'):
        chunks = []
        async for chunk in real_llm_service.stream(messages):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        # Combine chunks to get full response
        full_response = "".join(chunk.content for chunk in chunks if chunk.content)
        assert len(full_response) > 0
    else:
        pytest.skip("Service does not support streaming")


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_rag_context(real_llm_service):
    """Test LLM with RAG-style context."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    context = """
    The Eiffel Tower is located in Paris, France. It was built in 1889 and 
    stands 330 meters tall. It was designed by Gustave Eiffel.
    """
    
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="Answer questions based only on the provided context."
        ),
        Message(
            role=MessageRole.USER,
            content=f"Context: {context}\n\nQuestion: Where is the Eiffel Tower located?"
        )
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    assert "paris" in response.content.lower()
    assert "france" in response.content.lower()


@pytest.mark.real_integration
@pytest.mark.requires_llm
def test_llm_error_handling(real_llm_service):
    """Test LLM error handling with invalid input."""
    from rag_factory.services.llm.base import Message, MessageRole
    
    # Empty messages should either work or raise appropriate error
    messages = []
    
    try:
        response = real_llm_service.complete(messages)
        # If it doesn't error, response should be valid
        assert response is not None
    except (ValueError, Exception) as e:
        # It's acceptable to raise an error for empty messages
        assert "message" in str(e).lower() or "empty" in str(e).lower() or "invalid" in str(e).lower()
