"""
Agentic RAG strategy package.

This package provides an agent-based RAG strategy where an LLM agent
dynamically selects and uses appropriate retrieval tools based on query type.
"""

from .strategy import AgenticRAGStrategy
from .tools import Tool, ToolParameter, ToolResult
from .tool_implementations import (
    SemanticSearchTool,
    DocumentReaderTool,
    MetadataSearchTool,
    HybridSearchTool
)
from .agent import SimpleAgent, AgentState
from .query_analyzer import QueryAnalyzer, QueryType, QueryAnalysis
from .config import AgenticStrategyConfig

__all__ = [
    "AgenticRAGStrategy",
    "Tool",
    "ToolParameter",
    "ToolResult",
    "SemanticSearchTool",
    "DocumentReaderTool",
    "MetadataSearchTool",
    "HybridSearchTool",
    "SimpleAgent",
    "AgentState",
    "QueryAnalyzer",
    "QueryType",
    "QueryAnalysis",
    "AgenticStrategyConfig",
]
