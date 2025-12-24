"""
Agent implementation for agentic RAG strategy.

This module provides the core agent logic for selecting and executing
tools based on query analysis.
"""

from typing import List, Dict, Any, Optional
import logging

from .tools import Tool, ToolResult
from ...services.llm import LLMService
from ...services.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)


class AgentState:
    """State management for agent execution.
    
    Tracks the agent's progress through a query, including tool calls,
    results, and iteration count.
    """

    def __init__(self, max_iterations: int = 3):
        """Initialize agent state.
        
        Args:
            max_iterations: Maximum number of tool call iterations
        """
        self.query = ""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: List[ToolResult] = []
        self.iterations = 0
        self.max_iterations = max_iterations
        self.final_results = []

    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any]):
        """Record a tool call.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Parameters passed to the tool
        """
        self.tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "iteration": self.iterations
        })

    def add_tool_result(self, result: ToolResult):
        """Record a tool result.
        
        Args:
            result: Result from tool execution
        """
        self.tool_results.append(result)

    def should_continue(self) -> bool:
        """Check if agent should continue iterating.
        
        Returns:
            True if agent should continue, False otherwise
        """
        # Stop if max iterations reached
        if self.iterations >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return False
        
        # Continue if no results yet
        if not self.tool_results:
            return True
        
        # Check if we have sufficient successful results
        successful_results = [r for r in self.tool_results if r.success]
        if not successful_results:
            logger.info("No successful results yet, continuing")
            return True
        
        # Check if we have enough data
        total_chunks = sum(
            len(r.data) if isinstance(r.data, list) else (1 if r.data else 0)
            for r in successful_results
        )
        
        if total_chunks >= 5:  # Sufficient results
            logger.info(f"Sufficient results ({total_chunks} chunks), stopping")
            return False
        
        return True


class SimpleAgent:
    """
    Workflow-based agentic implementation using LLM for workflow selection.
    
    This agent uses an LLM to select from 6 predefined workflows,
    dramatically reducing context usage from 10K+ tokens to ~300 tokens.
    """

    def __init__(self, llm_service: LLMService, tools: List[Tool]):
        """Initialize agent.
        
        Args:
            llm_service: LLM service for workflow selection
            tools: List of available tools
        """
        self.llm_service = llm_service
        self.tools = {tool.name: tool for tool in tools}

    async def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the agent to retrieve information using workflow-based approach.
        
        Args:
            query: User query
            max_iterations: Not used in workflow-based approach (kept for compatibility)
            
        Returns:
            Dict with results and execution trace
        """
        from .workflows import WORKFLOWS, extract_plan_from_response, execute_workflow
        
        logger.info(f"Agent starting for query: {query}")

        # Select workflow using LLM
        plan_selection = self._select_workflow(query)
        plan_number = plan_selection["plan"]
        reasoning = plan_selection["reasoning"]
        
        logger.info(f"Selected workflow {plan_number}: {WORKFLOWS[plan_number]['name']}")
        logger.info(f"Reasoning: {reasoning}")

        # Execute the selected workflow
        try:
            workflow_results = await execute_workflow(plan_number, query, self.tools)
            
            # Synthesize final results
            final_results = self._synthesize_results(workflow_results)
            
            # Count successful steps
            successful_steps = sum(1 for r in workflow_results if r.success)
            
            logger.info(
                f"Workflow completed: {len(final_results)} results from "
                f"{successful_steps}/{len(workflow_results)} successful steps"
            )
            
            return {
                "results": final_results,
                "trace": {
                    "query": query,
                    "plan_number": plan_number,
                    "plan_name": WORKFLOWS[plan_number]["name"],
                    "reasoning": reasoning,
                    "steps_executed": len(workflow_results),
                    "steps_successful": successful_steps,
                    "workflow_results": [
                        {
                            "tool": r.tool_name,
                            "success": r.success,
                            "execution_time": r.execution_time,
                            "num_results": len(r.data) if isinstance(r.data, list) else (1 if r.data else 0),
                            "error": r.error
                        }
                        for r in workflow_results
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                "results": [],
                "trace": {
                    "query": query,
                    "plan_number": plan_number,
                    "error": str(e)
                }
            }

    def _select_workflow(self, query: str) -> Dict[str, Any]:
        """Select which workflow to use based on query.
        
        Args:
            query: User query
            
        Returns:
            Dict with "plan" (int 1-6) and "reasoning" (str)
        """
        from .workflows import WORKFLOWS, extract_plan_from_response
        
        # Build workflow selection prompt
        workflow_descriptions = []
        for num, workflow in WORKFLOWS.items():
            workflow_descriptions.append(
                f"{num}. {workflow['name']} - {workflow['description']}"
            )
        
        prompt = f"""Choose the best retrieval workflow for this query.

Query: {query}

Available Workflows:
{chr(10).join(workflow_descriptions)}

Respond in JSON format:
{{
  "plan": <number 1-6>,
  "reasoning": "<brief explanation>"
}}
"""

        messages = [Message(role=MessageRole.USER, content=prompt)]
        
        try:
            response = self.llm_service.complete(messages, temperature=0.3, max_tokens=200)
            return extract_plan_from_response(response.content)
        except Exception as e:
            logger.error(f"Workflow selection failed: {e}")
            # Fallback to plan 1 (simple semantic search)
            return {
                "plan": 1,
                "reasoning": f"Fallback due to error: {str(e)}"
            }

    def _synthesize_results(self, workflow_results: List[ToolResult]) -> List[Any]:
        """Combine and deduplicate results from workflow steps.
        
        Args:
            workflow_results: List of ToolResult from workflow execution
            
        Returns:
            List of deduplicated results
        """
        all_results = []
        seen_ids = set()

        for result in workflow_results:
            if not result.success:
                continue

            if isinstance(result.data, list):
                for item in result.data:
                    # Deduplicate by chunk_id or doc_id
                    item_id = item.get("chunk_id") or item.get("document_id") or str(item)
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        all_results.append(item)
            elif result.data:
                all_results.append(result.data)

        # Rank by relevance if we have scores
        if all_results and isinstance(all_results[0], dict) and "score" in all_results[0]:
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results

