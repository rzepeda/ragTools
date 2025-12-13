"""Integration tests for query expansion."""

import pytest
from rag_factory.strategies.query_expansion import (
    QueryExpanderService,
    ExpansionConfig,
    ExpansionStrategy
)


@pytest.mark.integration
class TestQueryExpansionIntegration:
    """Integration tests for query expansion with real LLM."""

    def test_keyword_expansion_real_llm(self, llm_service_from_env):
        """Test keyword expansion with real LLM."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            max_additional_terms=5,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        result = service.expand("machine learning")

        assert result.original_query == "machine learning"
        assert len(result.primary_expansion.expanded_query) > len(result.original_query)
        assert len(result.primary_expansion.added_terms) > 0
        assert result.execution_time_ms > 0

        print(f"\nOriginal: {result.original_query}")
        print(f"Expanded: {result.primary_expansion.expanded_query}")
        print(f"Added terms: {result.primary_expansion.added_terms}")

    def test_query_reformulation(self, llm_service_from_env):
        """Test query reformulation strategy."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.REFORMULATION,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        vague_query = "how does it work"
        result = service.expand(vague_query)

        # Reformulated query should be different
        assert result.primary_expansion.expanded_query != vague_query
        assert len(result.primary_expansion.expanded_query) > 0

        print(f"\nOriginal: {result.original_query}")
        print(f"Reformulated: {result.primary_expansion.expanded_query}")

    def test_question_generation(self, llm_service_from_env):
        """Test question generation strategy."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.QUESTION_GENERATION,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        result = service.expand("python tutorial")

        # Should generate a question
        assert result.primary_expansion.expanded_query != "python tutorial"
        # Questions typically contain question words or end with ?
        expanded = result.primary_expansion.expanded_query.lower()
        has_question_indicator = (
            "?" in result.primary_expansion.expanded_query or
            any(word in expanded for word in ["what", "how", "where", "when", "why", "who"])
        )
        assert has_question_indicator

        print(f"\nOriginal: {result.original_query}")
        print(f"Question: {result.primary_expansion.expanded_query}")

    def test_hyde_expansion(self, llm_service_from_env):
        """Test HyDE (Hypothetical Document Expansion)."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.HYDE,
            max_tokens=150,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        query = "What is the capital of France?"
        result = service.expand(query)

        # HyDE should generate a passage, not just keywords
        assert len(result.primary_expansion.expanded_query) > 50
        assert result.primary_expansion.expansion_strategy == ExpansionStrategy.HYDE

        print(f"\nQuery: {result.original_query}")
        print(f"Hypothetical document: {result.primary_expansion.expanded_query}")

    def test_multi_query_generation(self, llm_service_from_env):
        """Test generating multiple query variants."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.MULTI_QUERY,
            generate_multiple_variants=True,
            num_variants=3,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        result = service.expand("climate change effects")

        # Should have multiple variants
        assert len(result.expanded_queries) >= 1

        print(f"\nOriginal: {result.original_query}")
        print("Variants:")
        for i, variant in enumerate(result.expanded_queries, 1):
            print(f"  {i}. {variant.expanded_query}")

    def test_expansion_performance(self, llm_service_from_env):
        """Test expansion performance meets requirements."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        result = service.expand("artificial intelligence")

        print(f"\nExpansion took {result.execution_time_ms:.0f}ms")

        # Should meet <1 second requirement
        assert result.execution_time_ms < 1000, \
            f"Expansion took {result.execution_time_ms}ms (expected <1000ms)"

    def test_cache_functionality(self, llm_service_from_env):
        """Test that caching works correctly."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        query = "neural networks"

        # First call - cache miss
        result1 = service.expand(query)
        assert result1.cache_hit is False
        time1 = result1.execution_time_ms

        # Second call - cache hit
        result2 = service.expand(query)
        assert result2.cache_hit is True
        time2 = result2.execution_time_ms

        # Cached call should be much faster
        assert time2 < time1
        # Expanded queries should be the same
        assert result1.primary_expansion.expanded_query == result2.primary_expansion.expanded_query

        print(f"\nFirst call: {time1:.0f}ms (cache miss)")
        print(f"Second call: {time2:.0f}ms (cache hit)")

    def test_ab_testing_functionality(self, llm_service_from_env):
        """Test A/B testing capability."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            track_metrics=True,
            enable_cache=False  # Disable cache for this test
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        query = "neural networks"

        # Test with expansion enabled
        result_expanded = service.expand(query, enable_expansion=True)

        # Test with expansion disabled
        result_original = service.expand(query, enable_expansion=False)

        # Verify different results
        assert result_expanded.primary_expansion.expanded_query != result_original.primary_expansion.expanded_query
        assert len(result_expanded.primary_expansion.added_terms) > 0
        assert len(result_original.primary_expansion.added_terms) == 0

        # Check stats
        stats = service.get_stats()
        assert stats["expansion_enabled_count"] >= 1
        assert stats["expansion_disabled_count"] >= 1

        print(f"\nExpanded: {result_expanded.primary_expansion.expanded_query}")
        print(f"Original: {result_original.primary_expansion.expanded_query}")

    def test_domain_context(self, llm_service_from_env):
        """Test expansion with domain context."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            domain_context="medical and healthcare context",
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        result = service.expand("diagnosis")

        # Should expand with medical context
        assert result.primary_expansion.expanded_query != "diagnosis"

        print(f"\nOriginal: {result.original_query}")
        print(f"Expanded (medical context): {result.primary_expansion.expanded_query}")

    def test_error_handling_fallback(self, llm_service_from_env):
        """Test that service handles errors gracefully."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            enable_cache=False
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        # Force an error by using invalid query (empty after validation)
        # Actually, let's test with a normal query but break the service
        original_expander = service.expander
        service.expander = None  # This will cause an error

        result = service.expand("test query")

        # Should fallback to original query
        assert result.original_query == "test query"
        assert result.primary_expansion.expanded_query == "test query"
        assert "error" in result.metadata

        # Restore the expander
        service.expander = original_expander

    def test_stats_tracking(self, llm_service_from_env):
        """Test that statistics are tracked correctly."""
        expansion_config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            track_metrics=True,
            enable_cache=True
        )

        service = QueryExpanderService(expansion_config, llm_service_from_env)

        # Perform several expansions
        service.expand("query 1")
        service.expand("query 2")
        service.expand("query 1")  # Cache hit

        stats = service.get_stats()

        assert stats["total_expansions"] == 3
        assert stats["cache_hits"] >= 1
        assert stats["avg_execution_time_ms"] > 0
        assert stats["strategy"] == ExpansionStrategy.KEYWORD.value

        print(f"\nStats after 3 expansions:")
        print(f"  Total: {stats['total_expansions']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Avg time: {stats['avg_execution_time_ms']:.0f}ms")
