"""Data models for self-reflective RAG strategy."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class GradeLevel(Enum):
    """Grade levels for result quality."""
    EXCELLENT = 5  # Perfect match
    GOOD = 4       # Good match
    FAIR = 3       # Partial match
    POOR = 2       # Poor match
    IRRELEVANT = 1 # No match


class Grade(BaseModel):
    """Grade for a retrieved chunk.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        score: Grade score on 1-5 scale
        level: Grade level enum
        relevance: Relevance score 0-1
        completeness: Completeness score 0-1
        reasoning: Explanation for the grade
        metadata: Additional metadata
    """
    chunk_id: str
    score: float = Field(..., ge=1.0, le=5.0, description="Grade score 1-5")
    level: GradeLevel
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness score 0-1")
    reasoning: str = Field(..., description="Explanation for grade")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RefinementStrategy(Enum):
    """Query refinement strategies."""
    EXPANSION = "expansion"              # Add more keywords
    REFORMULATION = "reformulation"      # Rephrase query
    DECOMPOSITION = "decomposition"      # Break into sub-queries
    SPECIFICITY = "specificity"          # Adjust specificity
    CONTEXT_ADDITION = "context_addition" # Add domain context


class QueryRefinement(BaseModel):
    """Refined query generated from feedback.
    
    Attributes:
        original_query: The original user query
        refined_query: The refined version of the query
        strategy: Refinement strategy used
        reasoning: Explanation for the refinement
        iteration: Iteration number
        metadata: Additional metadata
    """
    original_query: str
    refined_query: str
    strategy: RefinementStrategy
    reasoning: str
    iteration: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalAttempt(BaseModel):
    """Single retrieval attempt with results and grades.
    
    Attributes:
        attempt_number: Attempt number (1-indexed)
        query: Query used for this attempt
        results: Retrieved results
        grades: Grades for each result
        average_grade: Average grade across all results
        refinement: Query refinement (if any)
        timestamp: Unix timestamp
        latency_ms: Latency in milliseconds
    """
    attempt_number: int
    query: str
    results: List[Dict[str, Any]]
    grades: List[Grade]
    average_grade: float
    refinement: Optional[QueryRefinement] = None
    timestamp: float
    latency_ms: float


class SelfReflectiveResult(BaseModel):
    """Complete self-reflective retrieval result.
    
    Attributes:
        original_query: Original user query
        final_results: Final aggregated results
        attempts: All retrieval attempts
        total_attempts: Total number of attempts
        success: Whether retrieval was successful
        final_average_grade: Final average grade
        total_latency_ms: Total latency in milliseconds
        metadata: Additional metadata
    """
    original_query: str
    final_results: List[Dict[str, Any]]
    attempts: List[RetrievalAttempt]
    total_attempts: int
    success: bool
    final_average_grade: float
    total_latency_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
