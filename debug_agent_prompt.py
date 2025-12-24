#!/usr/bin/env python3
"""Diagnostic to see what's in the agent's prompt."""

import sys
sys.path.insert(0, '/mnt/MCPProyects/ragTools')

from rag_factory.strategies.agentic.agent import SimpleAgent, AgentState
from rag_factory.strategies.agentic.tool_implementations import SemanticSearchTool

# Mock services
class MockLLM:
    def complete(self, messages, **kwargs):
        # Print the prompt
        print("=" * 60)
        print("PROMPT BEING SENT TO LLM:")
        print("=" * 60)
        for msg in messages:
            print(f"\nRole: {msg.role}")
            print(f"Content length: {len(msg.content)} chars")
            print(f"Content preview (first 500 chars):")
            print(msg.content[:500])
            print("...")
            if len(msg.content) > 500:
                print(f"\n[... {len(msg.content) - 500} more characters ...]")
        print("=" * 60)
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4
        print(f"\nTotal characters: {total_chars}")
        print(f"Estimated tokens: ~{estimated_tokens}")
        print("=" * 60)
        
        # Return mock response
        class MockResponse:
            content = "DONE"
            cost = 0.0
        return MockResponse()

class MockRepo:
    pass

class MockEmbedding:
    pass

# Create agent with mock services
tools = [SemanticSearchTool(MockRepo(), MockEmbedding())]
agent = SimpleAgent(MockLLM(), tools)

# Create state with a failed tool result
state = AgentState(max_iterations=3)
state.iterations = 1
state.query = "What is Voyager 1?"

# Simulate a failed tool result
from rag_factory.strategies.agentic.tools import ToolResult
failed_result = ToolResult(
    tool_name="semantic_search",
    success=False,
    data=None,
    error="Vector search failed: column 'text' does not exist",
    execution_time=0.1
)
state.add_tool_result(failed_result)

# Now call _select_tools to see the prompt
print("\nCalling _select_tools (iteration 2)...")
print()
agent._select_tools("What is Voyager 1?", state)
