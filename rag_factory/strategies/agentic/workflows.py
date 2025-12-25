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
from rag_factory.services.interfaces import ILLMService

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


async def substitute_params(
    params: Dict[str, Any],
    query: str,
    previous_results: List[ToolResult],
    llm_service: ILLMService,
    schema: Dict[str, str],
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
        llm_service: LLM service for metadata extraction.
        schema: Metadata schema for extraction.
        
    Returns:
        Parameter dict with substituted values
    """
    result = {}
    
    # Call LLM once to get all metadata
    extracted_metadata = await extract_metadata_from_query(query, schema, llm_service)

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
                    data_item = prev_result.data[0] if isinstance(prev_result.data, list) else prev_result.data
                    
                    logger.debug(f"Trying to extract {field_name} from step {step_num + 1} result")
                    
                    if isinstance(data_item, dict) and field_name in data_item:
                        result[key] = data_item[field_name]
                        logger.info(f"Successfully substituted {value} with {data_item[field_name]}")
                        continue
                    else:
                        logger.warning(f"Field {field_name} not found in step {step_num+1} result.")
                else:
                    logger.warning(f"Step {step_num + 1} failed or has no data")
            else:
                logger.warning(f"Step {step_num + 1} has not been executed yet")
            
            logger.warning(f"Could not substitute {value}, skipping parameter {key}")
            continue
        
        # Substitute {extracted_metadata}
        if "{extracted_metadata}" in value:
            metadata_filter = extracted_metadata.copy()
            metadata_filter.pop("document_id", None)
            result[key] = metadata_filter
            continue
        
        # Substitute {extracted_document_id}
        if "{extracted_document_id}" in value:
            result[key] = extracted_metadata.get("document_id")
            continue
        
        # No substitution needed
        result[key] = value
    
    return result


METADATA_EXTRACTION_PROMPT_TEMPLATE = """
You are a metadata extraction assistant. Extract structured metadata from the user query based on the provided schema.

USER QUERY:
"{query}"

METADATA SCHEMA:
{formatted_schema_fields}

INSTRUCTIONS:
1. Extract ONLY metadata that is explicitly mentioned or clearly implied in the query.
2. Match extracted values to the schema field names exactly.
3. Pay special attention to document identifiers, reference numbers, or document IDs (e.g., "DOC-123", "document 456", "report ABC-2023").
4. Return ONLY valid JSON with extracted fields.
5. If no metadata matches, return empty JSON: {{}}.
6. Do not include explanations, markdown formatting, or additional text.

OUTPUT FORMAT:
Return only the JSON object, nothing else.
"""


def _format_schema_for_prompt(schema: Dict[str, str]) -> str:
    """Formats the metadata schema for inclusion in the LLM prompt."""
    return "\n".join([f"- {name}: {description}" for name, description in schema.items()])


def _parse_llm_json_response(response: str) -> Dict[str, Any]:
    """
    Flexibly parses a JSON object from an LLM's text response.
    Handles markdown code blocks and other surrounding text.
    """
    if not response:
        return {}

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback to regex if direct parsing fails
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning(f"Could not parse extracted JSON from LLM response: {match.group(0)}")
    
    logger.warning(f"Could not parse JSON from LLM response: {response}")
    return {}


async def extract_metadata_from_query(
    query: str,
    schema: Dict[str, str],
    llm_service: ILLMService,
) -> Dict[str, Any]:
    """
    Extract metadata from natural language query using LLM-based parsing.

    Args:
        query: Natural language query string.
        schema: Dictionary mapping field names to descriptions.
            Example: {
                "document_id": "document identifier or reference",
                "year": "publication year",
                "author": "author or organization"
            }
        llm_service: LLM service instance for extraction.

    Returns:
        Dictionary of extracted metadata matching schema fields.
        Empty dict if no metadata found or extraction fails.
    
    Examples:
        query = "show me NASA report DOC-2023-001 from last year"
        schema = {"document_id": "document ID", "author": "organization"}
        # result = {"document_id": "DOC-2023-001", "author": "NASA"}

    Note:
        - Document IDs preserve exact format/casing as extracted.
        - Unmatched schema fields are omitted from result.
        - LLM failures return empty dict gracefully.
    """
    if not query or not schema:
        return {}

    formatted_schema = _format_schema_for_prompt(schema)
    prompt = METADATA_EXTRACTION_PROMPT_TEMPLATE.format(
        query=query, formatted_schema_fields=formatted_schema
    )

    try:
        response_text = await llm_service.complete_async(prompt, temperature=0.0, max_tokens=512)
        extracted_data = _parse_llm_json_response(response_text)

        if not isinstance(extracted_data, dict):
            logger.warning(f"LLM returned non-dict data: {type(extracted_data)}")
            return {}

        # Filter out keys not in schema to prevent hallucinated fields
        validated_data = {
            k: v for k, v in extracted_data.items() if k in schema
        }

        if len(validated_data) < len(extracted_data):
            removed_keys = set(extracted_data.keys()) - set(validated_data.keys())
            logger.warning(f"Removed hallucinated keys not in schema: {removed_keys}")
            
        return validated_data
    except Exception as e:
        logger.error(f"Error during LLM metadata extraction: {e}", exc_info=True)
        return {}


async def execute_workflow(
    plan_number: int,
    query: str,
    tools: Dict[str, Any],
    llm_service: ILLMService,
    schema: Dict[str, str],
) -> List[ToolResult]:
    """
    Execute a predefined workflow.
    
    Args:
        plan_number: Workflow number (1-6)
        query: User query
        tools: Dict of available tools {tool_name: tool_instance}
        llm_service: LLM service for metadata extraction.
        schema: Metadata schema for extraction.
        
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
        params = await substitute_params(params_template, query, results, llm_service, schema)
        
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
