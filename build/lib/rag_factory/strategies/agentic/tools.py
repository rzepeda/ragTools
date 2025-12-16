"""
Base classes and interfaces for agentic RAG tools.

This module defines the core abstractions for tools that agents can use
to retrieve information from the knowledge base.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import time


class ToolParameter(BaseModel):
    """Tool parameter definition.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, number, array, object, boolean)
        description: Human-readable description for LLM
        required: Whether parameter is required
        default: Default value if not required
    """
    name: str
    type: str  # "string", "integer", "number", "array", "object", "boolean"
    description: str
    required: bool = True
    default: Any = None


class ToolResult(BaseModel):
    """Result from tool execution.
    
    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        data: Result data (chunks, documents, etc.)
        error: Error message if execution failed
        execution_time: Time taken to execute in seconds
        metadata: Additional metadata about execution
    """
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Base class for agent tools.
    
    All tools must implement this interface to be usable by agents.
    Tools encapsulate specific retrieval operations and provide
    LLM-understandable descriptions and parameter schemas.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used as identifier).
        
        Returns:
            Unique tool name (e.g., "semantic_search")
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does.
        
        This description is shown to the LLM to help it decide
        when to use this tool. Be clear and specific.
        
        Returns:
            Human-readable description
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """Tool parameters.
        
        Returns:
            List of parameter definitions
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters as keyword arguments
            
        Returns:
            ToolResult with execution outcome
        """
        pass

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format.
        
        Converts this tool definition to the format expected by
        Anthropic's Claude API for tool use.
        
        Returns:
            Dictionary in Anthropic tool format
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }

    def _measure_execution(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Helper to measure execution time.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        return result, execution_time
