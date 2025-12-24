"""Workflow-based planning system for agentic RAG.

This module defines predefined workflows that the agent can choose from,
reducing LLM context usage from 10K+ tokens to ~300 tokens.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .tools import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    tool: str
    params: Dict[str, Any]


# Define the 6 predefined workflows
WORKFLOWS = {
    1: {
        "name": "Simple Semantic Search",
        "description": "For factual questions, 'What is...', 'How does...', 'Explain...'",
        "steps": [
            {
                "tool": "semantic_search",
                "params": {
                    "query": "{query}",
                    "top_k": 5,
                    "min_score": 0.7
                }
            }
        ]
    },
    
    2: {
        "name": "Metadata-Filtered Search",
        "description": "When query mentions attributes (author, date, source, category)",
        "steps": [
            {
                "tool": "metadata_search",
                "params": {
                    "query": "{query}",
                    "metadata_filter": "{extracted_metadata}",
                    "top_k": 5,
                    "min_score": 0.7
                }
            }
        ]
    },
    
    3: {
        "name": "Search Then Read Document",
        "description": "For comprehensive information or 'full document' requests",
        "steps": [
            {
                "tool": "semantic_search",
                "params": {
                    "query": "{query}",
                    "top_k": 1,
                    "min_score": 0.7
                }
            },
            {
                "tool": "read_document",
                "params": {
                    "document_id": "{from_step_1.document_id}"
                }
            }
        ]
    },
    
    4: {
        "name": "Hybrid Search",
        "description": "When query has both semantic intent and metadata hints",
        "steps": [
            {
                "tool": "hybrid_search",
                "params": {
                    "query": "{query}",
                    "top_k": 5,
                    "semantic_weight": 0.7
                }
            }
        ]
    },
    
    5: {
        "name": "Multi-Step Refinement",
        "description": "For broad exploratory queries",
        "steps": [
            {
                "tool": "semantic_search",
                "params": {
                    "query": "{query}",
                    "top_k": 3,
                    "min_score": 0.6
                }
            },
            {
                "tool": "hybrid_search",
                "params": {
                    "query": "{query}",
                    "metadata_hint": "{from_step_1.metadata}",
                    "top_k": 3
                }
            },
            {
                "tool": "read_document",
                "params": {
                    "document_id": "{from_step_1.document_id}"
                }
            }
        ]
    },
    
    6: {
        "name": "Direct Document Access",
        "description": "When document ID/name is explicitly mentioned",
        "steps": [
            {
                "tool": "read_document",
                "params": {
                    "document_id": "{extracted_document_id}"
                }
            }
        ]
    }
}


def extract_plan_from_response(response: Optional[str]) -> Dict[str, Any]:
    """
    Flexibly extract plan number from LLM response.
    
    Handles various response formats:
    - Clean JSON: {"plan": 3, "reasoning": "..."}
    - JSON with text: "I'll use plan 3. {\"plan\": 3, ...}"
    - Malformed: "plan: 3" or "I recommend workflow 4"
    - Just number: "5"
    - Invalid: defaults to plan 1
    
    Args:
        response: LLM response text (can be None or empty)
        
    Returns:
        Dict with "plan" (int 1-6) and "reasoning" (str)
    """
    if not response:
        return {
            "plan": 1,
            "reasoning": "Fallback: empty response, using simple semantic search"
        }
    
    # Strategy 1: Try to find and parse JSON block
    json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if "plan" in parsed:
                plan_num = int(parsed["plan"])
                if 1 <= plan_num <= 6:
                    return {
                        "plan": plan_num,
                        "reasoning": parsed.get("reasoning", "Extracted from JSON")
                    }
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    
    # Strategy 2: Look for "plan":<number> or "plan": <number> pattern
    plan_match = re.search(r'"?plan"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)
    if plan_match:
        try:
            plan_num = int(plan_match.group(1))
            if 1 <= plan_num <= 6:
                return {
                    "plan": plan_num,
                    "reasoning": "Extracted from plan field"
                }
        except ValueError:
            pass
    
    # Strategy 3: Look for "workflow <number>" pattern
    workflow_match = re.search(r'workflow\s+(\d+)', response, re.IGNORECASE)
    if workflow_match:
        try:
            plan_num = int(workflow_match.group(1))
            if 1 <= plan_num <= 6:
                return {
                    "plan": plan_num,
                    "reasoning": "Extracted from workflow mention"
                }
        except ValueError:
            pass
    
    # Strategy 4: Look for any number 1-6 in the response
    number_match = re.search(r'\b([1-6])\b', response)
    if number_match:
        return {
            "plan": int(number_match.group(1)),
            "reasoning": "Extracted number from response"
        }
    
    # Fallback: Default to plan 1 (simple semantic search)
    logger.warning(f"Could not parse plan from response: {response[:100]}...")
    return {
        "plan": 1,
        "reasoning": "Fallback: could not parse response, using simple semantic search"
    }


def substitute_params(
    params: Dict[str, Any],
    query: str,
    previous_results: List[ToolResult]
) -> Dict[str, Any]:
    """
    Substitute placeholders in parameters with actual values.
    
    Placeholders:
    - {query}: Replaced with the user's query
    - {from_step_1.field}: Replaced with field from step 1 results
    - {extracted_metadata}: Replaced with metadata extracted from query
    - {extracted_document_id}: Replaced with document ID extracted from query
    
    Args:
        params: Parameter dict with placeholders
        query: User query string
        previous_results: List of results from previous workflow steps
        
    Returns:
        Parameter dict with substituted values
    """
    result = {}
    
    for key, value in params.items():
        if not isinstance(value, str):
            result[key] = value
            continue
        
        # Substitute {query}
        if "{query}" in value:
            result[key] = value.replace("{query}", query)
            continue
        
        # Substitute {from_step_X.field}
        step_match = re.match(r'\{from_step_(\d+)\.(\w+)\}', value)
        if step_match:
            step_num = int(step_match.group(1)) - 1  # Convert to 0-indexed
            field_name = step_match.group(2)
            
            if step_num < len(previous_results):
                prev_result = previous_results[step_num]
                if prev_result.success and prev_result.data:
                    # Get first item from data if it's a list
                    data_item = prev_result.data[0] if isinstance(prev_result.data, list) else prev_result.data
                    
                    # Debug logging
                    logger.debug(f"Trying to extract {field_name} from step {step_num + 1} result")
                    logger.debug(f"Data item type: {type(data_item)}, keys: {data_item.keys() if isinstance(data_item, dict) else 'N/A'}")
                    
                    if isinstance(data_item, dict) and field_name in data_item:
                        result[key] = data_item[field_name]
                        logger.info(f"Successfully substituted {value} with {data_item[field_name]}")
                        continue
                    else:
                        logger.warning(f"Field {field_name} not found in data item. Available fields: {list(data_item.keys()) if isinstance(data_item, dict) else 'N/A'}")
                else:
                    logger.warning(f"Step {step_num + 1} failed or has no data")
            else:
                logger.warning(f"Step {step_num + 1} has not been executed yet")
            
            # If we couldn't substitute, log warning and skip this param
            # This will cause the tool to fail with missing required parameter
            logger.warning(f"Could not substitute {value}, skipping parameter {key}")
            continue
        
        # Substitute {extracted_metadata} - simple extraction from query
        if "{extracted_metadata}" in value:
            metadata = extract_metadata_from_query(query)
            result[key] = metadata
            continue
        
        # Substitute {extracted_document_id}
        if "{extracted_document_id}" in value:
            doc_id = extract_document_id_from_query(query)
            result[key] = doc_id
            continue
        
        # No substitution needed
        result[key] = value
    
    return result


def extract_metadata_from_query(query: str) -> Dict[str, Any]:
    """
    Extract metadata filters from natural language query.
    
    This is a simple heuristic-based extraction. In production,
    you might want to use an LLM for this.
    
    Args:
        query: User query string
        
    Returns:
        Dict of metadata filters
    """
    metadata = {}
    
    # Extract year/date patterns
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        metadata["date"] = year_match.group(0)
    
    # Extract common organizations (simple keyword matching)
    orgs = ["NASA", "ESA", "SpaceX", "JPL"]
    for org in orgs:
        if org.lower() in query.lower():
            metadata["author"] = org
            break
    
    # Extract document types
    doc_types = ["report", "paper", "specification", "manual", "guide"]
    for doc_type in doc_types:
        if doc_type in query.lower():
            metadata["type"] = doc_type
            break
    
    return metadata


def extract_document_id_from_query(query: str) -> Optional[str]:
    """
    Extract document ID from query.
    
    Looks for patterns like:
    - "document ABC-123"
    - "doc_12345"
    - "file://path/to/doc.pdf"
    
    Args:
        query: User query string
        
    Returns:
        Extracted document ID or None
    """
    # Pattern 1: document/doc followed by ID
    doc_match = re.search(r'(?:document|doc)\s+([A-Za-z0-9_-]+)', query, re.IGNORECASE)
    if doc_match:
        return doc_match.group(1)
    
    # Pattern 2: Standalone ID-like pattern
    id_match = re.search(r'\b([A-Za-z0-9]{8,})\b', query)
    if id_match:
        return id_match.group(1)
    
    return None


async def execute_workflow(
    plan_number: int,
    query: str,
    tools: Dict[str, Any]
) -> List[ToolResult]:
    """
    Execute a predefined workflow.
    
    Args:
        plan_number: Workflow number (1-6)
        query: User query
        tools: Dict of available tools {tool_name: tool_instance}
        
    Returns:
        List of ToolResult from each step
        
    Raises:
        KeyError: If workflow or tool not found
    """
    workflow = WORKFLOWS[plan_number]
    results = []
    
    logger.info(f"Executing workflow {plan_number}: {workflow['name']}")
    
    for step_idx, step in enumerate(workflow["steps"]):
        tool_name = step["tool"]
        params_template = step["params"]
        
        # Substitute parameters
        params = substitute_params(params_template, query, results)
        
        # Get tool
        if tool_name not in tools:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            results.append(ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=error_msg,
                execution_time=0.0
            ))
            break
        
        tool = tools[tool_name]
        
        # Execute tool
        try:
            logger.info(f"Executing step {step_idx + 1}: {tool_name} with params {params}")
            result = await tool.execute(**params)
            results.append(result)
            
            # Stop if tool failed
            if not result.success:
                logger.warning(f"Tool {tool_name} failed: {result.error}")
                break
                
        except Exception as e:
            logger.error(f"Tool {tool_name} raised exception: {e}")
            results.append(ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0
            ))
            break
    
    logger.info(f"Workflow completed with {len(results)} steps")
    return results
