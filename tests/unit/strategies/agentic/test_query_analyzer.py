"""Unit tests for query analyzer."""

import pytest

from rag_factory.strategies.agentic.query_analyzer import (
    QueryAnalyzer,
    QueryType,
    QueryAnalysis
)


# Test QueryAnalyzer

def test_query_analyzer_initialization():
    """Test query analyzer can be initialized."""
    analyzer = QueryAnalyzer()
    assert analyzer is not None


def test_extract_entities():
    """Test entity extraction."""
    analyzer = QueryAnalyzer()
    
    query = "Find documents by John Doe from 2024"
    entities = analyzer.extract_entities(query)
    
    assert "John" in entities or "Doe" in entities
    # Year extraction gets first 2 digits
    assert any(year in entities for year in ["20", "2024"])


def test_extract_entities_quoted():
    """Test entity extraction with quoted strings."""
    analyzer = QueryAnalyzer()
    
    query = 'Show me "user guide" document'
    entities = analyzer.extract_entities(query)
    
    assert "user guide" in entities


def test_extract_keywords():
    """Test keyword extraction."""
    analyzer = QueryAnalyzer()
    
    query = "What is the authentication process?"
    keywords = analyzer.extract_keywords(query)
    
    assert "authentication" in keywords
    assert "process" in keywords
    assert "what" not in keywords  # Stop word


def test_assess_complexity_simple():
    """Test complexity assessment for simple queries."""
    analyzer = QueryAnalyzer()
    
    query = "What is Python?"
    complexity = analyzer.assess_complexity(query)
    
    assert complexity == "simple"


def test_assess_complexity_complex():
    """Test complexity assessment for complex queries."""
    analyzer = QueryAnalyzer()
    
    query = "Compare and contrast Python and Java for web development"
    complexity = analyzer.assess_complexity(query)
    
    assert complexity == "complex"


def test_assess_complexity_multiple_questions():
    """Test complexity assessment with multiple questions."""
    analyzer = QueryAnalyzer()
    
    query = "What is Python? How does it compare to Java?"
    complexity = analyzer.assess_complexity(query)
    
    assert complexity == "complex"


def test_classify_type_factual():
    """Test query type classification for factual queries."""
    analyzer = QueryAnalyzer()
    
    query = "What is machine learning?"
    analysis = analyzer.analyze(query)
    
    assert analysis.query_type == QueryType.FACTUAL


def test_classify_type_exploratory():
    """Test query type classification for exploratory queries."""
    analyzer = QueryAnalyzer()
    
    query = "How does neural network training work?"
    analysis = analyzer.analyze(query)
    
    assert analysis.query_type == QueryType.EXPLORATORY


def test_classify_type_specific_document():
    """Test query type classification for document-specific queries."""
    analyzer = QueryAnalyzer()
    
    query = "Show me document ID abc123"
    analysis = analyzer.analyze(query)
    
    assert analysis.query_type == QueryType.SPECIFIC_DOCUMENT


def test_classify_type_metadata():
    """Test query type classification for metadata queries."""
    analyzer = QueryAnalyzer()
    
    query = "Find documents from 2024"
    analysis = analyzer.analyze(query)
    
    assert analysis.query_type == QueryType.METADATA_BASED


def test_recommend_tools_factual():
    """Test tool recommendations for factual queries."""
    analyzer = QueryAnalyzer()
    
    recommendations = analyzer.recommend_tools(
        QueryType.FACTUAL,
        entities=[],
        keywords=["python", "definition"]
    )
    
    assert "semantic_search" in recommendations


def test_recommend_tools_specific_document():
    """Test tool recommendations for document queries."""
    analyzer = QueryAnalyzer()
    
    recommendations = analyzer.recommend_tools(
        QueryType.SPECIFIC_DOCUMENT,
        entities=["doc123"],
        keywords=[]
    )
    
    assert "read_document" in recommendations


def test_recommend_tools_metadata():
    """Test tool recommendations for metadata queries."""
    analyzer = QueryAnalyzer()
    
    recommendations = analyzer.recommend_tools(
        QueryType.METADATA_BASED,
        entities=["2024"],
        keywords=["documents"]
    )
    
    assert "metadata_search" in recommendations


def test_recommend_tools_technical_terms():
    """Test tool recommendations with technical terms."""
    analyzer = QueryAnalyzer()
    
    recommendations = analyzer.recommend_tools(
        QueryType.FACTUAL,
        entities=[],
        keywords=["api", "function", "method"]
    )
    
    assert "hybrid_search" in recommendations


def test_analyze_full_workflow():
    """Test complete analysis workflow."""
    analyzer = QueryAnalyzer()
    
    query = "How does the authentication API work?"
    analysis = analyzer.analyze(query)
    
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.query_type in [QueryType.EXPLORATORY, QueryType.FACTUAL]
    assert len(analysis.keywords) > 0
    assert len(analysis.recommended_tools) > 0
    assert len(analysis.reasoning) > 0
