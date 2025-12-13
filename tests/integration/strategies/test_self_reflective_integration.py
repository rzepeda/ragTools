"""Integration tests for self-reflective RAG strategy."""

import pytest
from unittest.mock import Mock
from rag_factory.strategies.self_reflective import SelfReflectiveRAGStrategy


@pytest.mark.integration
class TestSelfReflectiveIntegration:
    """Integration tests for self-reflective RAG workflow."""

    @pytest.fixture
    def mock_base_strategy(self):
        """Create a mock base strategy."""
        strategy = Mock()
        strategy.retrieve.return_value = [
            {"chunk_id": "c1", "text": "Machine learning is a subset of AI", "score": 0.9},
            {"chunk_id": "c2", "text": "Deep learning uses neural networks", "score": 0.8},
            {"chunk_id": "c3", "text": "Python is a programming language", "score": 0.7}
        ]
        return strategy

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = Mock()
        
        # Mock grading response
        service.complete.side_effect = [
            # First call: grading
            Mock(content="""Result 1:
Grade: 3
Relevance: 0.6
Completeness: 0.5
Reasoning: Partial match, could be more complete

Result 2:
Grade: 3
Relevance: 0.6
Completeness: 0.5
Reasoning: Relevant but incomplete

Result 3:
Grade: 2
Relevance: 0.3
Completeness: 0.3
Reasoning: Not very relevant
"""),
            # Second call: refinement
            Mock(content="""Refined Query: What are the key concepts and applications of machine learning?
Strategy: expansion
Reasoning: Adding more context to get better results
"""),
            # Third call: grading again
            Mock(content="""Result 1:
Grade: 5
Relevance: 0.95
Completeness: 0.9
Reasoning: Excellent match, highly relevant

Result 2:
Grade: 4
Relevance: 0.85
Completeness: 0.8
Reasoning: Good match, relevant

Result 3:
Grade: 4
Relevance: 0.8
Completeness: 0.75
Reasoning: Good match
""")
        ]
        
        return service

    def test_end_to_end_workflow(self, mock_base_strategy, mock_llm_service):
        """Test complete self-reflective retrieval workflow."""
        # Create strategy
        strategy = SelfReflectiveRAGStrategy(
            base_retrieval_strategy=mock_base_strategy,
            llm_service=mock_llm_service,
            config={"grade_threshold": 4.0, "max_retries": 2}
        )

        # Retrieve with self-reflection
        results = strategy.retrieve("What is machine learning?", top_k=3)

        # Verify results
        assert len(results) > 0
        assert all("grade" in r for r in results)
        assert all("strategy" in r for r in results)
        assert all(r["strategy"] == "self_reflective" for r in results)
        
        # Should have triggered retry (poor initial grades)
        assert mock_base_strategy.retrieve.call_count == 2

    def test_retry_with_poor_results(self, mock_base_strategy, mock_llm_service):
        """Test that retry is triggered for poor results."""
        strategy = SelfReflectiveRAGStrategy(
            base_retrieval_strategy=mock_base_strategy,
            llm_service=mock_llm_service,
            config={"grade_threshold": 4.0, "max_retries": 2}
        )

        results = strategy.retrieve("test query", top_k=3)

        # Verify retry was triggered
        assert mock_base_strategy.retrieve.call_count >= 2
        
        # Verify refinements were made
        if results:
            assert "refinements" in results[0]

    def test_performance_within_limits(self, mock_base_strategy, mock_llm_service):
        """Test that self-reflective retrieval completes within timeout."""
        import time
        
        strategy = SelfReflectiveRAGStrategy(
            base_retrieval_strategy=mock_base_strategy,
            llm_service=mock_llm_service,
            config={"grade_threshold": 4.0, "max_retries": 2, "timeout_seconds": 10}
        )

        start = time.time()
        results = strategy.retrieve("test query", top_k=3)
        elapsed = time.time() - start

        # Should complete within timeout
        assert elapsed < 10.0
        assert len(results) > 0


@pytest.mark.integration
class TestSelfReflectiveWithLMStudio:
    """Integration tests with real LM Studio (uses .env configuration)."""

    def test_with_real_llm(self, llm_service_from_env):
        """Test with real LLM service from environment (LM Studio)."""
        from unittest.mock import Mock

        # Mock base strategy
        base_strategy = Mock()
        base_strategy.retrieve.return_value = [
            {"chunk_id": "c1", "text": "Sample result", "score": 0.9}
        ]

        # Create self-reflective strategy
        strategy = SelfReflectiveRAGStrategy(
            base_retrieval_strategy=base_strategy,
            llm_service=llm_service_from_env,
            config={"grade_threshold": 4.0, "max_retries": 1}
        )

        # Test retrieval
        results = strategy.retrieve("test query", top_k=1)

        assert len(results) > 0
        assert "grade" in results[0]
