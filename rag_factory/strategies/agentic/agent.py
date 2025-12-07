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
    Simple agentic implementation using LLM for tool selection.
    
    This agent uses an LLM to analyze queries and select appropriate
    tools for retrieval. It supports multi-step retrieval where results
    from one tool can inform the next tool selection.
    """

    def __init__(self, llm_service: LLMService, tools: List[Tool]):
        """Initialize agent.
        
        Args:
            llm_service: LLM service for tool selection
            tools: List of available tools
        """
        self.llm_service = llm_service
        self.tools = {tool.name: tool for tool in tools}
        self.tool_definitions = [tool.to_anthropic_tool() for tool in tools]

    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the agent to retrieve information.
        
        Args:
            query: User query
            max_iterations: Maximum tool call iterations
            
        Returns:
            Dict with results and execution trace
        """
        state = AgentState(max_iterations=max_iterations)
        state.query = query

        logger.info(f"Agent starting for query: {query}")

        # Planning phase: decide which tools to use
        plan = self._plan_retrieval(query)
        logger.info(f"Agent plan: {plan.get('reasoning', 'No reasoning provided')}")

        # Execution phase: run tools
        while state.should_continue():
            state.iterations += 1
            logger.info(f"Agent iteration {state.iterations}/{max_iterations}")

            # Get tool selection from LLM
            tool_calls = self._select_tools(query, state)

            if not tool_calls:
                logger.info("No more tools to call, stopping")
                break

            # Execute tools
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                state.add_tool_result(result)

                if result.success:
                    num_results = len(result.data) if isinstance(result.data, list) else 1
                    logger.info(
                        f"Tool {result.tool_name} succeeded in {result.execution_time:.2f}s, "
                        f"returned {num_results} results"
                    )
                else:
                    logger.warning(f"Tool {result.tool_name} failed: {result.error}")

        # Synthesis phase: combine results
        final_results = self._synthesize_results(state)

        return {
            "results": final_results,
            "trace": {
                "query": query,
                "iterations": state.iterations,
                "tool_calls": state.tool_calls,
                "tool_results": [
                    {
                        "tool": r.tool_name,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "num_results": len(r.data) if isinstance(r.data, list) else (1 if r.data else 0),
                        "error": r.error
                    }
                    for r in state.tool_results
                ],
                "plan": plan
            }
        }

    def _plan_retrieval(self, query: str) -> Dict[str, Any]:
        """Plan retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Dict with reasoning and cost
        """
        prompt = f"""Analyze this query and determine the best retrieval strategy:

Query: {query}

Available tools:
{self._format_tool_descriptions()}

Provide a brief plan for how to retrieve the information. Consider:
- What type of query is this? (factual, exploratory, specific document, metadata-based)
- Which tools would be most appropriate?
- Do you need multiple tools or multiple steps?

Keep your response concise (2-3 sentences)."""

        messages = [Message(role=MessageRole.USER, content=prompt)]
        
        try:
            response = self.llm_service.complete(messages, temperature=0.3, max_tokens=200)
            return {
                "reasoning": response.content,
                "cost": response.cost
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                "reasoning": "Planning failed, using default semantic search",
                "cost": 0.0
            }

    def _select_tools(self, query: str, state: AgentState) -> List[Dict[str, Any]]:
        """Select which tools to call.
        
        Args:
            query: User query
            state: Current agent state
            
        Returns:
            List of tool calls to execute
        """
        # Build context with previous results
        context = self._build_context(state)

        # Create prompt for tool selection
        prompt = f"""You are a retrieval agent. Select the appropriate tool to answer this query.

Query: {query}

{context}

Available tools:
{self._format_tool_descriptions()}

Based on the query and any previous results, which tool should we use next?
If we already have sufficient results, respond with "DONE".

Respond with ONLY the tool name and parameters in this format:
TOOL: tool_name
PARAMETERS: {{"param1": "value1", "param2": value2}}

Or respond with: DONE"""

        messages = [Message(role=MessageRole.USER, content=prompt)]

        try:
            response = self.llm_service.complete(messages, temperature=0.3, max_tokens=300)
            
            # Parse tool selections from response
            tool_calls = self._parse_tool_calls(response.content, query)
            
            for call in tool_calls:
                state.add_tool_call(call["tool"], call["parameters"])
            
            return tool_calls
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return []

    def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a tool.
        
        Args:
            tool_call: Dict with 'tool' and 'parameters' keys
            
        Returns:
            ToolResult from execution
        """
        tool_name = tool_call["tool"]
        parameters = tool_call["parameters"]

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Tool {tool_name} not found",
                execution_time=0.0
            )

        tool = self.tools[tool_name]
        try:
            return tool.execute(**parameters)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0
            )

    def _synthesize_results(self, state: AgentState) -> List[Any]:
        """Combine and deduplicate results from all tool calls.
        
        Args:
            state: Agent state with results
            
        Returns:
            List of deduplicated results
        """
        all_results = []
        seen_ids = set()

        for result in state.tool_results:
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

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt.
        
        Returns:
            Formatted string of tool descriptions
        """
        descriptions = []
        for tool in self.tools.values():
            params = ", ".join([
                f"{p.name}({p.type}{'*' if p.required else ''})"
                for p in tool.parameters
            ])
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)

    def _build_context(self, state: AgentState) -> str:
        """Build context string from previous results.
        
        Args:
            state: Agent state
            
        Returns:
            Context string
        """
        if not state.tool_results:
            return "This is the first retrieval step."

        context = f"Previous tool calls (iteration {state.iterations}):\n"
        for result in state.tool_results:
            if result.success:
                num_results = len(result.data) if isinstance(result.data, list) else 1
                context += f"- {result.tool_name}: {num_results} results found\n"
            else:
                context += f"- {result.tool_name}: failed ({result.error})\n"

        return context

    def _parse_tool_calls(self, response: str, query: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response.
        
        Args:
            response: LLM response text
            query: Original query (for fallback)
            
        Returns:
            List of tool call dicts
        """
        # Check for DONE signal
        if "DONE" in response.upper():
            return []

        tool_calls = []
        
        # Try to parse structured format
        lines = response.strip().split("\n")
        tool_name = None
        parameters = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("TOOL:"):
                tool_name = line.split(":", 1)[1].strip()
            elif line.startswith("PARAMETERS:"):
                param_str = line.split(":", 1)[1].strip()
                try:
                    import json
                    parameters = json.loads(param_str)
                except:
                    # Fallback: extract query parameter
                    parameters = {"query": query}
        
        if tool_name and tool_name in self.tools:
            # Ensure required parameters are present
            if "query" not in parameters:
                parameters["query"] = query
            tool_calls.append({
                "tool": tool_name,
                "parameters": parameters
            })
        else:
            # Fallback to semantic search
            logger.warning("Could not parse tool selection, falling back to semantic_search")
            tool_calls.append({
                "tool": "semantic_search",
                "parameters": {"query": query, "top_k": 5}
            })

        return tool_calls
