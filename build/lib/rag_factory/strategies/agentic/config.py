"""
Configuration for agentic RAG strategy.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class AgenticStrategyConfig(BaseModel):
    """Configuration for agentic RAG strategy.
    
    Attributes:
        max_iterations: Maximum number of tool call iterations (default: 3)
        enable_query_analysis: Whether to use query analysis (default: True)
        fallback_to_semantic: Fall back to semantic search on errors (default: True)
        timeout: Maximum time for retrieval in seconds (default: 30)
        enabled_tools: List of tool names to enable (default: all)
        llm_temperature: Temperature for LLM tool selection (default: 0.3)
        llm_max_tokens: Max tokens for LLM responses (default: 300)
    """
    max_iterations: int = Field(default=3, ge=1, le=10)
    enable_query_analysis: bool = True
    fallback_to_semantic: bool = True
    timeout: int = Field(default=30, ge=5, le=120)
    enabled_tools: Optional[List[str]] = None
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=300, ge=100, le=1000)
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields for extensibility
