"""Tests for workflow execution.

This module tests the execution of workflows with mocked tools.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.agentic.workflows import execute_workflow, substitute_params
from rag_factory.strategies.agentic.tools import ToolResult


@pytest.fixture
def mock_semantic_search_tool():
    """Mock semantic search tool."""
    tool = Mock()
    tool.name = "semantic_search"
    tool.execute = AsyncMock(return_value=ToolResult(
        tool_name="semantic_search",
        success=True,
        data=[
            {"chunk_id": "chunk1", "document_id": "doc1", "text": "Voyager 1 carries a golden record", "score": 0.95},
            {"chunk_id": "chunk2", "document_id": "doc1", "text": "The golden record contains sounds", "score": 0.89}
        ],
        error=None,
        execution_time=0.1
    ))
    return tool


@pytest.fixture
def mock_read_document_tool():
    """Mock read document tool."""
    tool = Mock()
    tool.name = "read_document"
    tool.execute = AsyncMock(return_value=ToolResult(
        tool_name="read_document",
        success=True,
        data=[
            {"chunk_id": f"chunk{i}", "text": f"Document chunk {i}"} 
            for i in range(10)
        ],
        error=None,
        execution_time=0.05
    ))
    return tool


@pytest.fixture
def mock_metadata_search_tool():
    """Mock metadata search tool."""
    tool = Mock()
    tool.name = "metadata_search"
    tool.execute = AsyncMock(return_value=ToolResult(
        tool_name="metadata_search",
        success=True,
        data=[
            {"chunk_id": "chunk3", "text": "NASA document from 1977", "score": 0.92}
        ],
        error=None,
        execution_time=0.12
    ))
    return tool


@pytest.fixture
def mock_tools(mock_semantic_search_tool, mock_read_document_tool, mock_metadata_search_tool):
    """Collection of all mock tools."""
    return {
        "semantic_search": mock_semantic_search_tool,
        "read_document": mock_read_document_tool,
        "metadata_search": mock_metadata_search_tool,
    }


class TestParameterSubstitution:
    """Test parameter substitution logic."""

    def test_substitute_query_placeholder(self):
        """Test that {query} is replaced with actual query."""
        params = {"query": "{query}", "top_k": 5}
        query = "What is Voyager 1?"
        
        result = substitute_params(params, query, [])
        
        assert result["query"] == "What is Voyager 1?"
        assert result["top_k"] == 5

    def test_substitute_from_step_placeholder(self):
        """Test that {from_step_1} is replaced with data from previous step."""
        params = {"document_id": "{from_step_1.document_id}"}
        query = "test"
        previous_results = [
            ToolResult(
                tool_name="semantic_search",
                success=True,
                data=[{"document_id": "doc123", "text": "content"}],
                error=None,
                execution_time=0.1
            )
        ]
        
        result = substitute_params(params, query, previous_results)
        
        assert result["document_id"] == "doc123"

    def test_substitute_no_placeholders(self):
        """Test params without placeholders remain unchanged."""
        params = {"top_k": 10, "threshold": 0.7}
        
        result = substitute_params(params, "query", [])
        
        assert result == params


@pytest.mark.asyncio
class TestWorkflowExecution:
    """Test execution of complete workflows."""

    async def test_execute_workflow_1_simple_search(self, mock_tools):
        """Test execution of workflow 1: Simple Semantic Search."""
        query = "What is the golden record?"
        
        results = await execute_workflow(1, query, mock_tools)
        
        assert len(results) == 1
        assert results[0].tool_name == "semantic_search"
        assert results[0].success is True
        assert len(results[0].data) > 0

    async def test_execute_workflow_3_search_then_read(self, mock_tools):
        """Test execution of workflow 3: Search Then Read Document."""
        query = "Tell me everything about Voyager"
        
        results = await execute_workflow(3, query, mock_tools)
        
        assert len(results) == 2
        assert results[0].tool_name == "semantic_search"
        assert results[1].tool_name == "read_document"
        # Verify chaining worked
        assert results[1].success is True

    async def test_execute_workflow_stops_on_failure(self, mock_tools):
        """Test that workflow stops if a tool fails."""
        # Make semantic_search fail
        mock_tools["semantic_search"].execute = AsyncMock(return_value=ToolResult(
            tool_name="semantic_search",
            success=False,
            data=None,
            error="Database connection failed",
            execution_time=0.1
        ))
        
        results = await execute_workflow(3, "test query", mock_tools)
        
        # Should only have 1 result (the failed one)
        assert len(results) == 1
        assert results[0].success is False
        # Second step should NOT have been executed
        assert not any(r.tool_name == "read_document" for r in results)

    async def test_execute_workflow_with_empty_results(self, mock_tools):
        """Test workflow execution when tool returns empty results."""
        # Make semantic_search return empty
        mock_tools["semantic_search"].execute = AsyncMock(return_value=ToolResult(
            tool_name="semantic_search",
            success=True,
            data=[],
            error=None,
            execution_time=0.1
        ))
        
        results = await execute_workflow(1, "test query", mock_tools)
        
        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].data) == 0

    async def test_execute_workflow_preserves_execution_order(self, mock_tools):
        """Test that multi-step workflows execute in correct order."""
        results = await execute_workflow(3, "test", mock_tools)
        
        # Results should be in order: semantic_search, then read_document
        assert results[0].tool_name == "semantic_search"
        assert results[1].tool_name == "read_document"

    async def test_execute_workflow_invalid_plan_number(self, mock_tools):
        """Test handling of invalid workflow number."""
        with pytest.raises(KeyError):
            await execute_workflow(99, "test", mock_tools)

    async def test_execute_workflow_metadata_extraction(self, mock_tools):
        """Test workflow 2 with metadata extraction."""
        query = "Show me NASA documents from 1977"
        
        results = await execute_workflow(2, query, mock_tools)
        
        assert len(results) == 1
        assert results[0].tool_name == "metadata_search"
        # Tool should have been called with extracted metadata
        call_args = mock_tools["metadata_search"].execute.call_args
        assert call_args is not None


class TestWorkflowErrorHandling:
    """Test error handling in workflow execution."""

    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_tools):
        """Test handling when a required tool is not available."""
        # Remove a tool
        del mock_tools["read_document"]
        
        with pytest.raises(KeyError):
            await execute_workflow(3, "test", mock_tools)

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self, mock_tools):
        """Test handling when tool raises an exception."""
        mock_tools["semantic_search"].execute = AsyncMock(
            side_effect=Exception("Database error")
        )
        
        results = await execute_workflow(1, "test", mock_tools)
        
        # Should capture exception and return failed result
        assert len(results) == 1
        assert results[0].success is False
        assert "error" in results[0].error.lower()


class TestWorkflowResultAggregation:
    """Test aggregation of results from multi-step workflows."""

    @pytest.mark.asyncio
    async def test_results_contain_all_steps(self, mock_tools):
        """Test that results list contains all executed steps."""
        results = await execute_workflow(3, "test", mock_tools)
        
        assert len(results) == 2
        assert all(isinstance(r, ToolResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_preserve_metadata(self, mock_tools):
        """Test that tool execution metadata is preserved."""
        results = await execute_workflow(1, "test", mock_tools)
        
        assert results[0].execution_time is not None
        assert results[0].tool_name is not None
