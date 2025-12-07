"""Unit tests for QueryVariantGenerator."""

import pytest
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.multi_query.variant_generator import QueryVariantGenerator
from rag_factory.strategies.multi_query.config import MultiQueryConfig


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock()
    # Mock async generate
    response = Mock()
    response.text = """What is machine learning?
How does machine learning work?
Explain machine learning concepts
Machine learning definition"""
    service.agenerate = AsyncMock(return_value=response)
    return service


@pytest.fixture
def config():
    """Create default config."""
    return MultiQueryConfig(num_variants=3)


@pytest.fixture
def variant_generator(mock_llm_service, config):
    """Create variant generator instance."""
    return QueryVariantGenerator(mock_llm_service, config)


@pytest.mark.asyncio
async def test_generate_variants_basic(variant_generator, mock_llm_service):
    """Test basic variant generation."""
    query = "What is machine learning?"

    variants = await variant_generator.generate_variants(query)

    # Should generate requested number of variants
    assert len(variants) >= 3
    assert query in variants  # Original included
    mock_llm_service.agenerate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_variants_include_original(mock_llm_service):
    """Test that original query is included when configured."""
    config = MultiQueryConfig(num_variants=3, include_original=True)
    generator = QueryVariantGenerator(mock_llm_service, config)

    query = "What is AI?"
    variants = await generator.generate_variants(query)

    assert query in variants


@pytest.mark.asyncio
async def test_generate_variants_exclude_original(mock_llm_service):
    """Test that original query can be excluded."""
    config = MultiQueryConfig(num_variants=3, include_original=False)
    generator = QueryVariantGenerator(mock_llm_service, config)

    query = "What is AI?"

    # Mock response without original query
    response = Mock()
    response.text = "How does AI work?\nExplain artificial intelligence\nAI definition"
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    variants = await generator.generate_variants(query)

    # Original should not be added
    # (might still be in parsed variants if LLM included it)
    assert len(variants) <= 4


@pytest.mark.asyncio
async def test_variant_validation(variant_generator, mock_llm_service):
    """Test variant validation removes invalid variants."""
    # Mock response with some invalid variants
    response = Mock()
    response.text = """Valid variant 1

    Valid variant 2
    short
    Valid variant 3"""
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    query = "Test query"
    variants = await variant_generator.generate_variants(query)

    # Should filter out empty lines and very short variants
    assert all(len(v) > 5 for v in variants if v != query)


@pytest.mark.asyncio
async def test_variant_generation_failure_fallback(mock_llm_service):
    """Test fallback to original query on generation failure."""
    config = MultiQueryConfig(fallback_to_original=True)
    generator = QueryVariantGenerator(mock_llm_service, config)

    # Mock LLM failure
    mock_llm_service.agenerate = AsyncMock(side_effect=Exception("LLM error"))

    query = "Test query"
    variants = await generator.generate_variants(query)

    # Should fall back to original query
    assert variants == [query]


@pytest.mark.asyncio
async def test_variant_deduplication(variant_generator, mock_llm_service):
    """Test that duplicate variants are removed."""
    # Mock response with duplicates
    response = Mock()
    response.text = """Variant 1
Variant 2
Variant 1
Variant 3"""
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    query = "Test"
    variants = await variant_generator.generate_variants(query)

    # Should remove duplicates (case-insensitive)
    assert len(variants) == len(set(v.lower() for v in variants))


@pytest.mark.asyncio
async def test_variant_generation_with_complete_method(config):
    """Test variant generation with LLM service using complete method."""
    # Mock LLM service with complete method instead of agenerate
    service = Mock()
    response = Mock()
    response.content = "Variant 1\nVariant 2\nVariant 3"
    service.complete = Mock(return_value=response)
    # Ensure no agenerate method
    if hasattr(service, 'agenerate'):
        delattr(service, 'agenerate')
    
    generator = QueryVariantGenerator(service, config)
    
    query = "Test query"
    variants = await generator.generate_variants(query)
    
    assert len(variants) >= 1
    # complete should have been called (via run_in_executor)
    assert service.complete.call_count >= 1
