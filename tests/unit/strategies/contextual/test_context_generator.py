"""
Unit tests for ContextGenerator.

Tests context generation functionality including LLM integration,
chunk selection, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from rag_factory.strategies.contextual.context_generator import ContextGenerator
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock()
    response = Mock()
    response.text = "This chunk discusses machine learning fundamentals in the context of an AI tutorial."
    service.agenerate = AsyncMock(return_value=response)
    return service


@pytest.fixture
def config():
    """Default configuration for testing."""
    return ContextualRetrievalConfig(
        context_length_min=20,  # Lower to match mock LLM response
        context_length_max=200,
        min_chunk_size_for_context=20  # Lower threshold for testing
    )


@pytest.fixture
def context_generator(mock_llm_service, config):
    """Context generator instance for testing."""
    return ContextGenerator(mock_llm_service, config)


@pytest.mark.asyncio
async def test_generate_context_basic(context_generator, mock_llm_service):
    """Test basic context generation."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Machine learning is a subset of AI that enables systems to learn from data. " * 5,
        "metadata": {"document_id": "doc_1"}
    }
    
    context = await context_generator.generate_context(chunk)
    
    assert context is not None
    assert len(context) > 0
    mock_llm_service.agenerate.assert_called_once()


@pytest.mark.asyncio
async def test_context_includes_document_metadata(context_generator, mock_llm_service):
    """Test that context uses document metadata."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Test text that is long enough to generate context for it properly. " * 5,
        "metadata": {
            "document_id": "doc_1",
            "section_hierarchy": ["Chapter 1", "Section 1.1"]
        }
    }
    
    document_context = {"title": "AI Tutorial"}
    
    context = await context_generator.generate_context(chunk, document_context)
    
    # Check that prompt included metadata
    call_args = mock_llm_service.agenerate.call_args
    prompt = call_args[1]["prompt"]
    
    assert "AI Tutorial" in prompt or "Section" in prompt


@pytest.mark.asyncio
async def test_skip_short_chunks(context_generator):
    """Test that very short chunks are skipped."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Short.",  # Very short
        "metadata": {}
    }
    
    context = await context_generator.generate_context(chunk)
    
    # Should skip due to length
    assert context is None


@pytest.mark.asyncio
async def test_skip_code_blocks(mock_llm_service):
    """Test skipping code blocks when configured."""
    config = ContextualRetrievalConfig(skip_code_blocks=True)
    generator = ContextGenerator(mock_llm_service, config)
    
    chunk = {
        "chunk_id": "chunk_1",
        "text": "```python\ndef hello():\n    print('Hello')\n```",
        "metadata": {}
    }
    
    context = await generator.generate_context(chunk)
    
    # Should skip code blocks
    assert context is None


@pytest.mark.asyncio
async def test_fallback_on_error(mock_llm_service):
    """Test fallback when context generation fails."""
    mock_llm_service.agenerate = AsyncMock(side_effect=Exception("LLM error"))
    
    config = ContextualRetrievalConfig(fallback_to_no_context=True)
    generator = ContextGenerator(mock_llm_service, config)
    
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Some text here that should be long enough for contextualization.",
        "metadata": {}
    }
    
    context = await generator.generate_context(chunk)
    
    # Should return None (fallback)
    assert context is None


@pytest.mark.asyncio
async def test_context_length_validation(mock_llm_service):
    """Test context length validation and truncation."""
    # Mock very long response
    response = Mock()
    response.text = "This is a very long context. " * 100  # Very long
    mock_llm_service.agenerate = AsyncMock(return_value=response)
    
    config = ContextualRetrievalConfig(context_length_max=200)
    generator = ContextGenerator(mock_llm_service, config)
    
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Test text that is long enough to generate context for it properly. " * 5,
        "metadata": {}
    }
    
    context = await generator.generate_context(chunk)
    
    # Should be truncated
    assert context is not None
    tokens = generator._count_tokens(context)
    assert tokens <= 250  # Allow some margin


@pytest.mark.asyncio
async def test_disabled_contextualization(context_generator):
    """Test that contextualization can be disabled."""
    context_generator.config.enable_contextualization = False
    
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Some text that would normally be contextualized.",
        "metadata": {}
    }
    
    context = await context_generator.generate_context(chunk)
    
    assert context is None


def test_token_counting(context_generator):
    """Test token counting approximation."""
    text = "This is a test text with some words."
    tokens = context_generator._count_tokens(text)
    
    # Should be approximately len(text) / 4
    assert tokens > 0
    assert tokens == len(text) // 4


def test_context_truncation(context_generator):
    """Test context truncation at sentence boundary."""
    long_text = "First sentence. Second sentence. Third sentence. " * 20
    max_tokens = 50
    
    truncated = context_generator._truncate_context(long_text, max_tokens)
    
    # Should be truncated
    assert len(truncated) <= max_tokens * 4 + 10  # Allow some margin
    # Should end with period or ellipsis
    assert truncated.endswith(".") or truncated.endswith("...")
