"""Unit tests for LLM-based query expander."""

import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.query_expansion.llm_expander import LLMQueryExpander
from rag_factory.strategies.query_expansion.base import (
    ExpansionConfig,
    ExpansionStrategy,
    ExpandedQuery
)
from rag_factory.services.llm.base import LLMResponse


@pytest.fixture
def expansion_config():
    """Create test expansion configuration."""
    return ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        max_additional_terms=5,
        temperature=0.3
    )


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = Mock()
    service.complete.return_value = LLMResponse(
        content="machine learning algorithms neural networks deep learning",
        model="gpt-3.5-turbo",
        provider="openai",
        prompt_tokens=50,
        completion_tokens=15,
        total_tokens=65,
        cost=0.0001,
        latency=0.5,
        metadata={}
    )
    return service


class TestLLMQueryExpander:
    """Tests for LLMQueryExpander class."""

    def test_initialization(self, expansion_config, mock_llm_service):
        """Test LLM expander initializes correctly."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        assert expander.config == expansion_config
        assert expander.llm_service == mock_llm_service
        assert expander.prompts is not None

    def test_expand_query(self, expansion_config, mock_llm_service):
        """Test query expansion."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        result = expander.expand("machine learning")

        assert result.original_query == "machine learning"
        assert result.expanded_query != ""
        assert result.expansion_strategy == ExpansionStrategy.KEYWORD
        assert isinstance(result.added_terms, list)
        assert "llm_model" in result.metadata
        mock_llm_service.complete.assert_called_once()

    def test_expand_calls_llm_with_correct_params(self, expansion_config, mock_llm_service):
        """Test that LLM is called with correct parameters."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        expander.expand("test query")

        call_args = mock_llm_service.complete.call_args
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["max_tokens"] == 150

    def test_validate_query_empty(self, expansion_config, mock_llm_service):
        """Test validation rejects empty query."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            expander.expand("")

    def test_validate_query_too_long(self, expansion_config, mock_llm_service):
        """Test validation rejects overly long query."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        long_query = "x" * 1001
        with pytest.raises(ValueError, match="Query too long"):
            expander.expand(long_query)

    def test_extract_added_terms(self, expansion_config, mock_llm_service):
        """Test extraction of added terms."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        result = expander.expand("machine learning")

        # Check that added terms were extracted
        assert len(result.added_terms) > 0
        # Original terms should not be in added_terms
        assert "machine" not in result.added_terms
        assert "learning" not in result.added_terms

    def test_expand_with_domain_context(self, mock_llm_service):
        """Test expansion with domain context."""
        config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            domain_context="medical domain"
        )
        expander = LLMQueryExpander(config, mock_llm_service)

        expander.expand("diagnosis")

        # Verify domain context was included in system prompt
        call_args = mock_llm_service.complete.call_args
        messages = call_args.kwargs["messages"]
        system_message = messages[0]
        assert "medical domain" in system_message.content

    def test_different_expansion_strategies(self, mock_llm_service):
        """Test different expansion strategies use different prompts."""
        strategies = [
            ExpansionStrategy.KEYWORD,
            ExpansionStrategy.REFORMULATION,
            ExpansionStrategy.QUESTION_GENERATION
        ]

        for strategy in strategies:
            config = ExpansionConfig(strategy=strategy)
            expander = LLMQueryExpander(config, mock_llm_service)

            result = expander.expand("test query")

            assert result.expansion_strategy == strategy

    def test_metadata_includes_token_info(self, expansion_config, mock_llm_service):
        """Test that metadata includes token usage information."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        result = expander.expand("test query")

        assert "prompt_tokens" in result.metadata
        assert "completion_tokens" in result.metadata
        assert "total_tokens" in result.metadata
        assert "cost" in result.metadata
        assert result.metadata["prompt_tokens"] == 50
        assert result.metadata["completion_tokens"] == 15
        assert result.metadata["total_tokens"] == 65

    def test_llm_response_trimmed(self, expansion_config, mock_llm_service):
        """Test that LLM response is trimmed of whitespace."""
        mock_llm_service.complete.return_value = LLMResponse(
            content="  test expanded query  \n",
            model="gpt-3.5-turbo",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.00001,
            latency=0.3,
            metadata={}
        )

        expander = LLMQueryExpander(expansion_config, mock_llm_service)
        result = expander.expand("test")

        assert result.expanded_query == "test expanded query"

    def test_confidence_is_one(self, expansion_config, mock_llm_service):
        """Test that confidence is set to 1.0."""
        expander = LLMQueryExpander(expansion_config, mock_llm_service)

        result = expander.expand("test query")

        assert result.confidence == 1.0
