"""Unit tests for ResultGrader."""

import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.self_reflective.grader import ResultGrader
from rag_factory.strategies.self_reflective.models import GradeLevel


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock()
    # Mock grading response
    service.complete.return_value = Mock(
        content="""Result 1:
Grade: 4
Relevance: 0.8
Completeness: 0.7
Reasoning: Good match, relevant to query

Result 2:
Grade: 2
Relevance: 0.3
Completeness: 0.4
Reasoning: Poor match, not very relevant
"""
    )
    return service


@pytest.fixture
def grader(mock_llm_service):
    """Create a ResultGrader instance."""
    config = {"batch_grading_size": 5}
    return ResultGrader(mock_llm_service, config)


def test_grade_results(grader, mock_llm_service):
    """Test grading results."""
    results = [
        {"chunk_id": "chunk1", "text": "Result 1 text"},
        {"chunk_id": "chunk2", "text": "Result 2 text"}
    ]

    grades = grader.grade_results("test query", results)

    assert len(grades) == 2
    assert grades[0].score == 4.0
    assert grades[0].level == GradeLevel.GOOD
    assert grades[1].score == 2.0
    assert grades[1].level == GradeLevel.POOR

    mock_llm_service.complete.assert_called_once()


def test_grade_parsing(grader):
    """Test parsing grades from LLM response."""
    response = """Result 1:
Grade: 5
Relevance: 1.0
Completeness: 1.0
Reasoning: Perfect match

Result 2:
Grade: 1
Relevance: 0.1
Completeness: 0.2
Reasoning: Not relevant
"""

    results = [
        {"chunk_id": "c1", "text": "text1"},
        {"chunk_id": "c2", "text": "text2"}
    ]

    grades = grader._parse_grades(response, results)

    assert len(grades) == 2
    assert grades[0].score == 5.0
    assert grades[0].level == GradeLevel.EXCELLENT
    assert grades[1].score == 1.0
    assert grades[1].level == GradeLevel.IRRELEVANT


def test_batch_grading(grader, mock_llm_service):
    """Test batch grading multiple results."""
    results = [
        {"chunk_id": f"chunk{i}", "text": f"Text {i}"}
        for i in range(10)
    ]

    grades = grader.grade_results("test query", results)

    # Should have called LLM twice (batch_size = 5)
    assert mock_llm_service.complete.call_count == 2
    assert len(grades) == 10


def test_grade_fallback_on_parse_error(grader):
    """Test fallback grade when parsing fails."""
    response = "Invalid response format"

    results = [{"chunk_id": "c1", "text": "text"}]

    grades = grader._parse_grades(response, results)

    # Should return fallback grade
    assert len(grades) == 1
    assert grades[0].score == 3.0  # Default middle grade
    assert grades[0].level == GradeLevel.FAIR


def test_grade_levels():
    """Test grade level assignment."""
    from rag_factory.strategies.self_reflective.models import Grade, GradeLevel
    
    # Test EXCELLENT (>= 4.5)
    grade = Grade(
        chunk_id="c1", score=5.0, level=GradeLevel.EXCELLENT,
        relevance=1.0, completeness=1.0, reasoning="test"
    )
    assert grade.level == GradeLevel.EXCELLENT
    
    # Test GOOD (>= 3.5)
    grade = Grade(
        chunk_id="c2", score=4.0, level=GradeLevel.GOOD,
        relevance=0.8, completeness=0.8, reasoning="test"
    )
    assert grade.level == GradeLevel.GOOD


def test_empty_results(grader):
    """Test grading empty results list."""
    grades = grader.grade_results("test query", [])
    assert len(grades) == 0


def test_grading_error_handling(mock_llm_service):
    """Test error handling when LLM fails."""
    # Configure mock to raise exception
    mock_llm_service.complete.side_effect = Exception("LLM error")
    
    grader = ResultGrader(mock_llm_service, {})
    results = [{"chunk_id": "c1", "text": "text"}]
    
    # Should return fallback grades instead of crashing
    grades = grader.grade_results("test query", results)
    
    assert len(grades) == 1
    assert grades[0].score == 3.0
    assert "failed" in grades[0].reasoning.lower()


def test_build_grading_prompt(grader):
    """Test grading prompt construction."""
    results = [
        {"chunk_id": "c1", "text": "Sample text 1"},
        {"chunk_id": "c2", "text": "Sample text 2"}
    ]
    
    prompt = grader._build_grading_prompt("test query", results)
    
    assert "test query" in prompt
    assert "Sample text 1" in prompt
    assert "Sample text 2" in prompt
    assert "1-5" in prompt
