"""Unit tests for agentic RAG agent."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rag_factory.strategies.agentic.agent import SimpleAgent, AgentState
from rag_factory.strategies.agentic.tools import Tool, ToolResult, ToolParameter


# Fixtures

@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = Mock()
    response = Mock()
    response.content = "I will use semantic_search to find relevant information"
    response.cost = 0.001
    service.complete.return_value = response
    return service


@pytest.fixture
def mock_tool():
    """Mock tool."""
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool description"
    tool.parameters = [
        ToolParameter(name="query", type="string", description="Query", required=True)
    ]
    tool.to_anthropic_tool.return_value = {
        "name": "test_tool",
        "description": "Test tool"
    }
    tool.execute.return_value = ToolResult(
        tool_name="test_tool",
        success=True,
        data=[{"chunk_id": "1", "text": "Result"}],
        execution_time=0.1
    )
    return tool


# Test AgentState

def test_agent_state_initialization():
    """Test agent state initialization."""
    state = AgentState()
    
    assert state.query == ""
    assert state.iterations == 0
    assert state.max_iterations == 3
    assert len(state.tool_calls) == 0
    assert len(state.tool_results) == 0


def test_agent_state_add_tool_call():
    """Test adding tool call to state."""
    state = AgentState()
    state.add_tool_call("semantic_search", {"query": "test"})
    
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0]["tool"] == "semantic_search"
    assert state.tool_calls[0]["parameters"]["query"] == "test"


def test_agent_state_add_tool_result():
    """Test adding tool result to state."""
    state = AgentState()
    result = ToolResult(
        tool_name="test",
        success=True,
        data=[],
        execution_time=0.1
    )
    state.add_tool_result(result)
    
    assert len(state.tool_results) == 1
    assert state.tool_results[0].tool_name == "test"


def test_agent_state_should_continue_max_iterations():
    """Test agent state stops at max iterations."""
    state = AgentState(max_iterations=2)
    
    assert state.should_continue() is True
    
    state.iterations = 2
    assert state.should_continue() is False


def test_agent_state_should_continue_sufficient_results():
    """Test agent state stops with sufficient results."""
    state = AgentState()
    
    # Add successful results
    for i in range(6):
        result = ToolResult(
            tool_name="test",
            success=True,
            data=[{"chunk_id": str(i), "text": f"Result {i}"}],
            execution_time=0.1
        )
        state.add_tool_result(result)
    
    assert state.should_continue() is False


# Test SimpleAgent

def test_agent_initialization(mock_llm_service, mock_tool):
    """Test agent initialization."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    assert len(agent.tools) == 1
    assert "test_tool" in agent.tools
    assert len(agent.tool_definitions) == 1


def test_agent_run_basic(mock_llm_service, mock_tool):
    """Test basic agent run."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    # Mock tool selection to stop after one iteration
    with patch.object(agent, '_select_tools', side_effect=[
        [{"tool": "test_tool", "parameters": {"query": "test"}}],
        []  # Return empty to stop
    ]):
        result = agent.run("test query", max_iterations=2)
    
    assert "results" in result
    assert "trace" in result
    assert result["trace"]["query"] == "test query"
    assert result["trace"]["iterations"] >= 1


def test_agent_planning_phase(mock_llm_service, mock_tool):
    """Test agent planning phase."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    plan = agent._plan_retrieval("test query")
    
    assert "reasoning" in plan
    assert "cost" in plan
    mock_llm_service.complete.assert_called()


def test_agent_tool_execution(mock_llm_service, mock_tool):
    """Test agent executes tools correctly."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    tool_call = {"tool": "test_tool", "parameters": {"query": "test"}}
    result = agent._execute_tool(tool_call)
    
    assert result.success is True
    assert result.tool_name == "test_tool"
    mock_tool.execute.assert_called_once()


def test_agent_tool_execution_not_found(mock_llm_service, mock_tool):
    """Test agent handles tool not found."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    tool_call = {"tool": "nonexistent_tool", "parameters": {}}
    result = agent._execute_tool(tool_call)
    
    assert result.success is False
    assert "not found" in result.error


def test_agent_synthesize_results(mock_llm_service, mock_tool):
    """Test result synthesis and deduplication."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    state = AgentState()
    
    # Add results with duplicates
    state.add_tool_result(ToolResult(
        tool_name="tool1",
        success=True,
        data=[
            {"chunk_id": "1", "text": "Result 1", "score": 0.9},
            {"chunk_id": "2", "text": "Result 2", "score": 0.8}
        ],
        execution_time=0.1
    ))
    state.add_tool_result(ToolResult(
        tool_name="tool2",
        success=True,
        data=[
            {"chunk_id": "2", "text": "Result 2", "score": 0.85},  # Duplicate
            {"chunk_id": "3", "text": "Result 3", "score": 0.7}
        ],
        execution_time=0.1
    ))
    
    results = agent._synthesize_results(state)
    
    # Should deduplicate chunk_id 2
    assert len(results) == 3
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates
    
    # Should be sorted by score
    assert results[0]["score"] >= results[1]["score"]


def test_agent_synthesize_results_ignores_failed(mock_llm_service, mock_tool):
    """Test synthesis ignores failed results."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    state = AgentState()
    state.add_tool_result(ToolResult(
        tool_name="tool1",
        success=False,
        data=[],
        error="Failed",
        execution_time=0.1
    ))
    state.add_tool_result(ToolResult(
        tool_name="tool2",
        success=True,
        data=[{"chunk_id": "1", "text": "Result"}],
        execution_time=0.1
    ))
    
    results = agent._synthesize_results(state)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"


def test_agent_max_iterations(mock_llm_service, mock_tool):
    """Test agent respects max iterations."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    # Mock to always return tool calls
    with patch.object(agent, '_select_tools', return_value=[
        {"tool": "test_tool", "parameters": {"query": "test"}}
    ]):
        result = agent.run("test query", max_iterations=2)
    
    assert result["trace"]["iterations"] <= 2


def test_agent_parse_tool_calls_structured(mock_llm_service, mock_tool):
    """Test parsing structured tool call response."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    response = """TOOL: test_tool
PARAMETERS: {"query": "test query", "top_k": 5}"""
    
    tool_calls = agent._parse_tool_calls(response, "test query")
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "test_tool"
    assert tool_calls[0]["parameters"]["query"] == "test query"


def test_agent_parse_tool_calls_done(mock_llm_service, mock_tool):
    """Test parsing DONE response."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])
    
    response = "DONE"
    tool_calls = agent._parse_tool_calls(response, "test query")
    
    assert len(tool_calls) == 0


def test_agent_parse_tool_calls_fallback(mock_llm_service):
    """Test fallback when parsing fails."""
    semantic_tool = Mock(spec=Tool)
    semantic_tool.name = "semantic_search"
    semantic_tool.description = "Semantic search"
    semantic_tool.parameters = []
    semantic_tool.to_anthropic_tool.return_value = {}
    
    agent = SimpleAgent(mock_llm_service, [semantic_tool])
    
    response = "Invalid response format"
    tool_calls = agent._parse_tool_calls(response, "test query")
    
    # Should fallback to semantic_search
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "semantic_search"
