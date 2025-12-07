"""Unit tests for QueryRefiner."""

import pytest
from unittest.mock import Mock
from rag_factory.strategies.self_reflective.refiner import QueryRefiner
from rag_factory.strategies.self_reflective.models import (
    Grade, GradeLevel, RefinementStrategy, QueryRefinement
)


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock()
    service.complete.return_value = Mock(
        content="""Refined Query: What are the main components of machine learning systems?
Strategy: expansion
Reasoning: Adding more specific keywords to get more relevant results
"""
    )
    return service


@pytest.fixture
def refiner(mock_llm_service):
    """Create a QueryRefiner instance."""
    config = {"refinement_strategy": RefinementStrategy.REFORMULATION}
    return QueryRefiner(mock_llm_service, config)


def test_refine_query(refiner, mock_llm_service):
    """Test query refinement."""
    grades = [
        Grade(
            chunk_id="c1",
            score=2.5,
            level=GradeLevel.FAIR,
            relevance=0.5,
            completeness=0.4,
            reasoning="Incomplete results"
        )
    ]

    refinement = refiner.refine_query(
        original_query="What is ML?",
        grades=grades,
        iteration=1
    )

    assert refinement.refined_query != "What is ML?"
    assert refinement.strategy in RefinementStrategy
    assert refinement.reasoning != ""
    assert refinement.iteration == 1

    mock_llm_service.complete.assert_called_once()


def test_identify_gaps(refiner):
    """Test gap identification from grades."""
    grades = [
        Grade(
            chunk_id="c1",
            score=2.0,
            level=GradeLevel.POOR,
            relevance=0.3,
            completeness=0.4,
            reasoning="Results are incomplete and not relevant enough"
        ),
        Grade(
            chunk_id="c2",
            score=3.0,
            level=GradeLevel.FAIR,
            relevance=0.6,
            completeness=0.5,
            reasoning="Query may be too vague"
        )
    ]

    gaps = refiner._identify_gaps(grades)

    assert len(gaps) > 0
    assert any("incomplete" in gap.lower() for gap in gaps)


def test_parse_refinement(refiner):
    """Test parsing refinement from LLM response."""
    response = """Refined Query: How do neural networks learn from data?
Strategy: reformulation
Reasoning: Rephrased to be more specific and clear
"""

    refined_query, strategy, reasoning = refiner._parse_refinement(response)

    assert refined_query == "How do neural networks learn from data?"
    assert strategy == RefinementStrategy.REFORMULATION
    assert "specific" in reasoning.lower()


def test_refinement_with_previous_attempts(refiner, mock_llm_service):
    """Test refinement avoids previous attempts."""
    previous = [
        QueryRefinement(
            original_query="What is AI?",
            refined_query="What is artificial intelligence?",
            strategy=RefinementStrategy.EXPANSION,
            reasoning="Added full term",
            iteration=1
        )
    ]

    grades = [
        Grade(
            chunk_id="c1",
            score=2.0,
            level=GradeLevel.POOR,
            relevance=0.3,
            completeness=0.3,
            reasoning="Not relevant"
        )
    ]

    refinement = refiner.refine_query(
        original_query="What is AI?",
        grades=grades,
        iteration=2,
        previous_refinements=previous
    )

    # Should not repeat previous refinement
    assert refinement.refined_query != "What is artificial intelligence?"


def test_refinement_strategies():
    """Test different refinement strategies."""
    strategies = [
        "expansion",
        "reformulation",
        "decomposition",
        "specificity",
        "context_addition"
    ]
    
    for strategy_str in strategies:
        strategy = RefinementStrategy(strategy_str)
        assert strategy.value == strategy_str


def test_empty_grades(refiner):
    """Test refinement with empty grades list."""
    gaps = refiner._identify_gaps([])
    assert len(gaps) > 0  # Should have at least one gap message


def test_refinement_error_handling(mock_llm_service):
    """Test error handling when LLM fails."""
    mock_llm_service.complete.side_effect = Exception("LLM error")
    
    refiner = QueryRefiner(mock_llm_service, {})
    grades = [
        Grade(
            chunk_id="c1", score=2.0, level=GradeLevel.POOR,
            relevance=0.3, completeness=0.3, reasoning="Poor"
        )
    ]
    
    # Should return fallback refinement instead of crashing
    refinement = refiner.refine_query("test query", grades, 1)
    
    assert refinement.refined_query is not None
    assert "failed" in refinement.reasoning.lower()


def test_build_refinement_prompt(refiner):
    """Test refinement prompt construction."""
    gaps = ["Results are incomplete", "Query too vague"]
    previous = [
        QueryRefinement(
            original_query="test",
            refined_query="test refined",
            strategy=RefinementStrategy.EXPANSION,
            reasoning="test",
            iteration=1
        )
    ]
    
    prompt = refiner._build_refinement_prompt("test query", gaps, previous)
    
    assert "test query" in prompt
    assert "incomplete" in prompt.lower()
    assert "vague" in prompt.lower()
    assert "test refined" in prompt
