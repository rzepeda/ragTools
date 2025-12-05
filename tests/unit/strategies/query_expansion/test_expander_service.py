"""Unit tests for query expander service."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_factory.strategies.query_expansion.expander_service import QueryExpanderService
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
        enable_cache=True
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


class TestQueryExpanderService:
    """Tests for QueryExpanderService class."""

    def test_service_initialization(self, expansion_config, mock_llm_service):
        """Test service initializes correctly."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        assert service.config == expansion_config
        assert service.llm_service == mock_llm_service
        assert service.expander is not None
        assert service.cache is not None

    def test_service_initialization_without_cache(self, mock_llm_service):
        """Test service initialization without cache."""
        config = ExpansionConfig(enable_cache=False)
        service = QueryExpanderService(config, mock_llm_service)

        assert service.cache is None

    def test_expand_basic(self, expansion_config, mock_llm_service):
        """Test basic query expansion."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        result = service.expand("machine learning")

        assert result.original_query == "machine learning"
        assert result.primary_expansion.expanded_query != ""
        assert len(result.expanded_queries) >= 1
        assert result.cache_hit is False
        assert result.llm_used == expansion_config.llm_model

    def test_expand_with_cache_hit(self, expansion_config, mock_llm_service):
        """Test expansion with cache hit."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        # First call - cache miss
        result1 = service.expand("test query")
        assert result1.cache_hit is False

        # Second call - cache hit
        result2 = service.expand("test query")
        assert result2.cache_hit is True

        # LLM should only be called once
        assert mock_llm_service.complete.call_count == 1

    def test_expand_disabled(self, expansion_config, mock_llm_service):
        """Test expansion when disabled."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        result = service.expand("test query", enable_expansion=False)

        # Should return original query
        assert result.original_query == "test query"
        assert result.primary_expansion.expanded_query == "test query"
        assert len(result.primary_expansion.added_terms) == 0
        assert "expansion_disabled" in result.metadata

        # LLM should not be called
        assert mock_llm_service.complete.call_count == 0

    def test_expand_with_error_fallback(self, expansion_config, mock_llm_service):
        """Test fallback to original query on error."""
        mock_llm_service.complete.side_effect = Exception("LLM error")

        service = QueryExpanderService(expansion_config, mock_llm_service)

        result = service.expand("test query")

        # Should fallback to original query
        assert result.original_query == "test query"
        assert result.primary_expansion.expanded_query == "test query"
        assert "error" in result.metadata

    def test_multiple_variants(self, mock_llm_service):
        """Test generating multiple query variants."""
        mock_llm_service.complete.return_value = LLMResponse(
            content="query variant 1\nquery variant 2\nquery variant 3",
            model="gpt-3.5-turbo",
            provider="openai",
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            cost=0.0001,
            latency=0.6,
            metadata={}
        )

        config = ExpansionConfig(
            strategy=ExpansionStrategy.MULTI_QUERY,
            generate_multiple_variants=True,
            num_variants=3
        )

        service = QueryExpanderService(config, mock_llm_service)

        result = service.expand("original query")

        assert len(result.expanded_queries) >= 1
        # Should have multiple variants for multi-query strategy
        if result.expanded_queries[0].expansion_strategy == ExpansionStrategy.MULTI_QUERY:
            assert len(result.expanded_queries) > 1

    def test_get_stats(self, expansion_config, mock_llm_service):
        """Test statistics tracking."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        service.expand("query 1")
        service.expand("query 1")  # Cache hit
        service.expand("query 2")
        service.expand("query 3", enable_expansion=False)

        stats = service.get_stats()

        assert stats["total_expansions"] == 4
        assert stats["cache_hits"] >= 1
        assert stats["expansion_enabled_count"] == 3
        assert stats["expansion_disabled_count"] == 1
        assert "cache_hit_rate" in stats
        assert "expansion_rate" in stats
        assert stats["strategy"] == ExpansionStrategy.KEYWORD.value

    def test_clear_cache(self, expansion_config, mock_llm_service):
        """Test cache clearing."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        # Add something to cache
        service.expand("query 1")
        service.expand("query 1")  # Should be cache hit

        # Clear cache
        service.clear_cache()

        # Next call should be cache miss
        result = service.expand("query 1")
        # This is the third call total, but first after clear
        assert mock_llm_service.complete.call_count >= 2

    def test_execution_time_tracked(self, expansion_config, mock_llm_service):
        """Test that execution time is tracked."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        result = service.expand("test query")

        assert result.execution_time_ms > 0

    def test_stats_average_execution_time(self, expansion_config, mock_llm_service):
        """Test average execution time calculation."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        service.expand("query 1")
        service.expand("query 2")
        service.expand("query 3")

        stats = service.get_stats()

        assert stats["avg_execution_time_ms"] > 0

    def test_cache_key_computation(self, expansion_config, mock_llm_service):
        """Test cache key computation is consistent."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        key1 = service._compute_cache_key("test query")
        key2 = service._compute_cache_key("test query")
        key3 = service._compute_cache_key("different query")

        assert key1 == key2
        assert key1 != key3

    def test_passthrough_result_metadata(self, expansion_config, mock_llm_service):
        """Test passthrough result has correct metadata."""
        service = QueryExpanderService(expansion_config, mock_llm_service)

        result = service.expand("test", enable_expansion=False)

        assert result.primary_expansion.metadata.get("passthrough") is True
        assert result.metadata.get("expansion_disabled") is True

    def test_hyde_expander_used_for_hyde_strategy(self, mock_llm_service):
        """Test that HyDE expander is used for HYDE strategy."""
        config = ExpansionConfig(strategy=ExpansionStrategy.HYDE)
        service = QueryExpanderService(config, mock_llm_service)

        from rag_factory.strategies.query_expansion.hyde_expander import HyDEExpander
        assert isinstance(service.expander, HyDEExpander)

    def test_llm_expander_used_for_other_strategies(self, mock_llm_service):
        """Test that LLM expander is used for non-HYDE strategies."""
        strategies = [
            ExpansionStrategy.KEYWORD,
            ExpansionStrategy.REFORMULATION,
            ExpansionStrategy.QUESTION_GENERATION
        ]

        from rag_factory.strategies.query_expansion.llm_expander import LLMQueryExpander

        for strategy in strategies:
            config = ExpansionConfig(strategy=strategy)
            service = QueryExpanderService(config, mock_llm_service)
            assert isinstance(service.expander, LLMQueryExpander)
