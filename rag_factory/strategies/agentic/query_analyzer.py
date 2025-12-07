"""
Query analysis for agentic RAG.

This module provides query analysis capabilities to help the agent
select appropriate tools based on query characteristics.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import logging
import re

from ...services.llm import LLMService
from ...services.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for classification."""
    FACTUAL = "factual"  # "What is X?", "Define Y"
    EXPLORATORY = "exploratory"  # "How does X work?", "Explain Y"
    SPECIFIC_DOCUMENT = "specific_document"  # "Show me document X", "Read file Y"
    METADATA_BASED = "metadata_based"  # "Find documents from 2024", "Papers by author X"
    HYBRID = "hybrid"  # Combination of above


class QueryAnalysis:
    """Result of query analysis."""
    
    def __init__(
        self,
        query_type: QueryType,
        entities: List[str],
        keywords: List[str],
        complexity: str,
        recommended_tools: List[str],
        reasoning: str
    ):
        """Initialize query analysis.
        
        Args:
            query_type: Classified query type
            entities: Extracted entities
            keywords: Key terms
            complexity: "simple" or "complex"
            recommended_tools: Suggested tools to use
            reasoning: Explanation of analysis
        """
        self.query_type = query_type
        self.entities = entities
        self.keywords = keywords
        self.complexity = complexity
        self.recommended_tools = recommended_tools
        self.reasoning = reasoning


class QueryAnalyzer:
    """Analyzes queries to guide tool selection."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize query analyzer.
        
        Args:
            llm_service: Optional LLM service for advanced analysis
        """
        self.llm_service = llm_service

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query to determine characteristics.
        
        Args:
            query: User query to analyze
            
        Returns:
            QueryAnalysis with classification and recommendations
        """
        # Extract basic features
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        query_type = self._classify_type(query)
        complexity = self.assess_complexity(query)
        
        # Recommend tools based on analysis
        recommended_tools = self.recommend_tools(query_type, entities, keywords)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query_type, complexity, recommended_tools)
        
        return QueryAnalysis(
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            complexity=complexity,
            recommended_tools=recommended_tools,
            reasoning=reasoning
        )

    def extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query.
        
        Args:
            query: User query
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction using patterns
        entities = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                entities.append(word)
        
        # Look for years
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        entities.extend(years)
        
        return list(set(entities))

    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'this', 'that'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords

    def assess_complexity(self, query: str) -> str:
        """Assess query complexity.
        
        Args:
            query: User query
            
        Returns:
            "simple" or "complex"
        """
        # Indicators of complexity
        complex_indicators = [
            'and', 'or', 'but', 'however', 'compare', 'contrast',
            'relationship', 'between', 'multiple', 'various'
        ]
        
        query_lower = query.lower()
        
        # Check for multiple questions
        if query.count('?') > 1:
            return "complex"
        
        # Check for complex indicators
        if any(indicator in query_lower for indicator in complex_indicators):
            return "complex"
        
        # Check length
        if len(query.split()) > 15:
            return "complex"
        
        return "simple"

    def recommend_tools(
        self,
        query_type: QueryType,
        entities: List[str],
        keywords: List[str]
    ) -> List[str]:
        """Recommend tools based on query analysis.
        
        Args:
            query_type: Classified query type
            entities: Extracted entities
            keywords: Key terms
            
        Returns:
            List of recommended tool names
        """
        recommendations = []
        
        if query_type == QueryType.SPECIFIC_DOCUMENT:
            recommendations.append("read_document")
        elif query_type == QueryType.METADATA_BASED:
            recommendations.append("metadata_search")
        elif query_type == QueryType.HYBRID:
            recommendations.extend(["hybrid_search", "semantic_search"])
        else:
            # Default to semantic search for factual/exploratory
            recommendations.append("semantic_search")
        
        # Add hybrid search if we have specific technical terms
        technical_terms = ['api', 'function', 'class', 'method', 'algorithm']
        if any(term in ' '.join(keywords) for term in technical_terms):
            if "hybrid_search" not in recommendations:
                recommendations.append("hybrid_search")
        
        return recommendations

    def _classify_type(self, query: str) -> QueryType:
        """Classify query type.
        
        Args:
            query: User query
            
        Returns:
            QueryType classification
        """
        query_lower = query.lower()
        
        # Check for document-specific queries
        doc_patterns = [
            'show me document', 'read document', 'get document',
            'document id', 'file named', 'show me the file'
        ]
        if any(pattern in query_lower for pattern in doc_patterns):
            return QueryType.SPECIFIC_DOCUMENT
        
        # Check for metadata queries
        metadata_patterns = [
            'from 20', 'in 20', 'by author', 'tagged with',
            'category', 'written by', 'published in'
        ]
        if any(pattern in query_lower for pattern in metadata_patterns):
            return QueryType.METADATA_BASED
        
        # Check for exploratory queries
        exploratory_patterns = [
            'how does', 'how do', 'explain', 'describe',
            'what are the steps', 'walk me through'
        ]
        if any(pattern in query_lower for pattern in exploratory_patterns):
            return QueryType.EXPLORATORY
        
        # Check for factual queries
        factual_patterns = [
            'what is', 'what are', 'define', 'who is',
            'when was', 'where is'
        ]
        if any(pattern in query_lower for pattern in factual_patterns):
            return QueryType.FACTUAL
        
        # Default to factual
        return QueryType.FACTUAL

    def _generate_reasoning(
        self,
        query_type: QueryType,
        complexity: str,
        recommended_tools: List[str]
    ) -> str:
        """Generate reasoning for analysis.
        
        Args:
            query_type: Classified type
            complexity: Complexity assessment
            recommended_tools: Recommended tools
            
        Returns:
            Reasoning string
        """
        return (
            f"Query classified as {query_type.value} with {complexity} complexity. "
            f"Recommended tools: {', '.join(recommended_tools)}"
        )
