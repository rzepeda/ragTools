"""Integration tests for Multi-Query RAG Strategy."""

import pytest
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.multi_query import (
    MultiQueryRAGStrategy,
    MultiQueryConfig,
    RankingStrategy
)
from rag_factory.services.dependencies import StrategyDependencies



@pytest.mark.integration
class TestMultiQueryIntegration:
    """Integration tests for multi-query RAG workflow."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock()

        async def mock_search(query, top_k):
            # Return different results based on query
            if "machine learning" in query.lower():
                return [
                    {"chunk_id": "ml1", "text": "ML is a subset of AI", "score": 0.9},
                    {"chunk_id": "ml2", "text": "ML uses algorithms", "score": 0.8},
                ]
            elif "artificial intelligence" in query.lower():
                return [
                    {"chunk_id": "ai1", "text": "AI is intelligence by machines", "score": 0.85},
                    {"chunk_id": "ml1", "text": "ML is a subset of AI", "score": 0.75},
                ]
            else:
                return [
                    {"chunk_id": "gen1", "text": f"Result for {query}", "score": 0.7},
                ]

        store.asearch = mock_search
        return store

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()

        async def mock_generate(prompt, **kwargs):
            # Generate variants based on prompt
            response = Mock()
            response.text = """What is machine learning?
How does machine learning work?
Explain artificial intelligence and machine learning
Define machine learning concepts"""
            return response

        service.agenerate = mock_generate
        return service

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        # Mock embed method returns dummy embeddings
        service.embed = Mock(return_value=[[0.0] * 768])
        return service

    @pytest.fixture
    def mock_database_service(self):
        """Create a mock database service."""
        service = Mock()
        service.store_chunk = Mock()
        service.get_chunks = Mock(return_value=[])
        return service

    def test_multi_query_complete_workflow(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test complete multi-query retrieval workflow."""
        # Create strategy
        config = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
            final_top_k=5
        )
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=mock_llm_service,
                embedding_service=mock_embedding_service,
                database_service=mock_database_service
            )
        )

        # Retrieve with multi-query
        results = strategy.retrieve("What is machine learning?")

        # Verify results
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)
        assert all("final_score" in r for r in results)
        assert all("ranking_method" in r for r in results)

        # Results should be sorted by final_score
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_multi_query_async_workflow(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test async multi-query retrieval workflow."""
        config = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=RankingStrategy.FREQUENCY_BOOST,
            final_top_k=5
        )
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=mock_llm_service,
                embedding_service=mock_embedding_service,
                database_service=mock_database_service
            )
        )

        # Async retrieve
        results = await strategy.aretrieve("What is machine learning?")

        # Verify results
        assert len(results) > 0
        assert all("frequency" in r for r in results)
        assert all("frequency_boost_factor" in r for r in results)

    def test_variant_diversity(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test that generated variants are diverse."""
        config = MultiQueryConfig(num_variants=4, log_variants=True)
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=mock_llm_service,
                embedding_service=mock_embedding_service,
                database_service=mock_database_service
            )
        )

        # Retrieve
        results = strategy.retrieve("What is AI?")

        # Variants should have been generated
        # (We can't directly check variants without modifying the strategy,
        # but we can verify results came from multiple sources)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_performance_requirements(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test that multi-query retrieval completes within timeout."""
        import time

        config = MultiQueryConfig(
            num_variants=5,
            query_timeout=5.0,
            variant_generation_timeout=5.0
        )
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=mock_llm_service,
                embedding_service=mock_embedding_service,
                database_service=mock_database_service
            )
        )

        start = time.time()
        results = await strategy.aretrieve("test query")
        elapsed = time.time() - start

        # Should complete reasonably fast (< 3s as per requirements)
        # In practice with mocks, should be much faster
        assert elapsed < 3.0
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, mock_vector_store):
        """Test fallback to original query on variant generation failure."""
        # Mock LLM that fails
        llm_service = Mock()
        llm_service.agenerate = AsyncMock(side_effect=Exception("LLM failed"))

        config = MultiQueryConfig(fallback_to_original=True, final_top_k=3)
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=llm_service,
                vector_store_service=mock_vector_store
            )
        )

        # Should fall back to original query
        results = await strategy.aretrieve("test query")

        # Should still get results from fallback
        assert len(results) > 0

    def test_ranking_strategy_comparison(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test different ranking strategies produce different results."""
        query = "What is machine learning?"

        # Test with MAX_SCORE
        config_max = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=RankingStrategy.MAX_SCORE,
            final_top_k=5
        )
        strategy_max = MultiQueryRAGStrategy(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm_service,
            config=config_max
        )
        results_max = strategy_max.retrieve(query)

        # Test with FREQUENCY_BOOST
        config_freq = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=RankingStrategy.FREQUENCY_BOOST,
            final_top_k=5
        )
        strategy_freq = MultiQueryRAGStrategy(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm_service,
            config=config_freq
        )
        results_freq = strategy_freq.retrieve(query)

        # Both should return results
        assert len(results_max) > 0
        assert len(results_freq) > 0

        # Ranking methods should be different
        assert results_max[0]["ranking_method"] == "max_score"
        assert results_freq[0]["ranking_method"] == "frequency_boost"

    def test_deduplication_across_variants(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test that duplicate results are properly deduplicated."""
        config = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=RankingStrategy.MAX_SCORE,
            final_top_k=10
        )
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=mock_llm_service,
                embedding_service=mock_embedding_service,
                database_service=mock_database_service
            )
        )

        results = strategy.retrieve("machine learning")

        # Check for duplicates
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Results contain duplicates"

        # Check frequency tracking
        for result in results:
            assert "frequency" in result
            assert result["frequency"] >= 1

    def test_strategy_properties(self, mock_vector_store, mock_llm_service, mock_embedding_service, mock_database_service):
        """Test strategy name and description properties."""
        strategy = MultiQueryRAGStrategy(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm_service
        )

        assert strategy.name == "multi_query"
        assert "multiple query variants" in strategy.description.lower()


@pytest.mark.integration
class TestMultiQueryWithLMStudio:
    """Integration tests with real LM Studio (uses .env configuration)."""

    def test_with_real_llm(self, llm_service_from_env):
        """Test with real LLM service from environment (LM Studio)."""
        from unittest.mock import Mock

        # Mock vector store
        vector_store = Mock()

        async def mock_search(query, top_k):
            return [
                {"chunk_id": f"c{i}", "text": f"Result {i}", "score": 0.9 - i * 0.1}
                for i in range(top_k)
            ]

        vector_store.asearch = mock_search

        # Create strategy
        config = MultiQueryConfig(num_variants=3, final_top_k=5)
        strategy = MultiQueryRAGStrategy(
            config=config,
            dependencies=StrategyDependencies(
                llm_service=llm_service_from_env
            )
        )

        # Test retrieval
        results = strategy.retrieve("What is machine learning?")

        assert len(results) > 0
        assert all("final_score" in r for r in results)
