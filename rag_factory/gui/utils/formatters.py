"""Formatting utilities for GUI display.

This module provides functions to format data for display in the GUI,
including YAML configurations, retrieval results, and error messages.
"""

import yaml
from typing import List, Dict, Any
from datetime import datetime


def format_yaml(config: Dict[str, Any]) -> str:
    """Format a configuration dictionary as YAML for display.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Formatted YAML string
        
    Example:
        >>> config = {"name": "test", "value": 123}
        >>> print(format_yaml(config))
        name: test
        value: 123
    """
    try:
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    except Exception as e:
        return f"Error formatting YAML: {e}"


def format_results(results: List[Any], show_metadata: bool = True) -> str:
    """Format retrieval results for display.
    
    Args:
        results: List of retrieved chunks/results
        show_metadata: Whether to show metadata fields
        
    Returns:
        Formatted string for display in results textbox
        
    Example:
        >>> results = [
        ...     {"score": 0.89, "content": "Machine learning...", "source": "ml.txt"},
        ...     {"score": 0.76, "content": "Types of ML...", "source": "ml.txt"}
        ... ]
        >>> print(format_results(results))
    """
    if not results:
        return "No results found."
    
    formatted_lines = []
    
    for i, result in enumerate(results, 1):
        # Handle different result formats
        if hasattr(result, '__dict__'):
            # Object with attributes
            score = getattr(result, 'score', None) or getattr(result, 'similarity', 0.0)
            content = getattr(result, 'content', None) or getattr(result, 'text', '')
            source = getattr(result, 'source', None) or getattr(result, 'document_id', 'Unknown')
            metadata = getattr(result, 'metadata', {})
        elif isinstance(result, dict):
            # Dictionary format
            score = result.get('score') or result.get('similarity', 0.0)
            content = result.get('content') or result.get('text', '')
            source = result.get('source') or result.get('document_id', 'Unknown')
            metadata = result.get('metadata', {})
        else:
            # Unknown format
            formatted_lines.append(f"{i}. {str(result)}\n")
            continue
        
        # Format result
        formatted_lines.append(f"{i}. Score: {score:.4f}")
        
        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."
        
        formatted_lines.append(f"   {content}")
        formatted_lines.append(f"   Source: {source}")
        
        # Add metadata if requested
        if show_metadata and metadata:
            formatted_lines.append(f"   Metadata: {metadata}")
        
        formatted_lines.append("")  # Empty line between results
    
    return "\n".join(formatted_lines)


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """Format an error message for display.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include full traceback
        
    Returns:
        Formatted error message
        
    Example:
        >>> try:
        ...     raise ValueError("Invalid input")
        ... except Exception as e:
        ...     print(format_error(e))
        Error: Invalid input
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if include_traceback:
        import traceback
        tb = traceback.format_exc()
        return f"{error_type}: {error_msg}\n\nTraceback:\n{tb}"
    else:
        return f"{error_type}: {error_msg}"


def format_timestamp(timestamp: datetime = None) -> str:
    """Format a timestamp for display.
    
    Args:
        timestamp: Datetime to format (defaults to now)
        
    Returns:
        Formatted timestamp string
        
    Example:
        >>> format_timestamp()
        '2025-12-18 23:57:54'
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds for display.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Example:
        >>> format_duration(0.345)
        '0.35s'
        >>> format_duration(65.2)
        '1m 5.20s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
