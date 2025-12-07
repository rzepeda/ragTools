"""Unit tests for agentic RAG tools."""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from rag_factory.strategies.agentic.tools import (
    Tool, ToolParameter, ToolResult
)
from rag_factory.strategies.agentic.tool_implementations import (
    SemanticSearchTool,
    DocumentReaderTool,
    MetadataSearchTool,
    HybridSearchTool
)


# Fixtures

@pytest.fixture
def mock_chunk_repository():
    """Mock chunk repository."""
    repo = Mock()
    
    # Mock search_similar
    repo.search_similar.return_value = [
        (Mock(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Result 1",
            metadata_={"key": "value"},
            chunk_index=0
        ), 0.9),
        (Mock(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Result 2",
            metadata_={},
            chunk_index=1
        ), 0.8)
    ]
    
    # Mock search_similar_with_metadata
    repo.search_similar_with_metadata.return_value = [
        (Mock(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Metadata result",
            metadata_={"author": "John"},
            chunk_index=0
        ), 0.85)
    ]
    
    # Mock get_by_document
    repo.get_by_document.return_value = [
        Mock(
            chunk_id=uuid4(),
            chunk_index=0,
            text="Chunk 1",
            metadata_={}
        ),
        Mock(
            chunk_id=uuid4(),
            chunk_index=1,
            text="Chunk 2",
            metadata_={}
        )
    ]
    
    return repo


@pytest.fixture
def mock_document_repository():
    """Mock document repository."""
    repo = Mock()
    
    doc_id = uuid4()
    repo.get_by_id.return_value = Mock(
        document_id=doc_id,
        filename="test.pdf",
        source_path="/path/to/test.pdf",
        metadata_={"author": "Test"},
        total_chunks=2,
        status="completed"
    )
    
    return repo


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.embed_text.return_value = [0.1] * 384  # Mock embedding vector
    return service


# Test Tool Base Classes

def test_tool_parameter_creation():
    """Test ToolParameter can be created."""
    param = ToolParameter(
        name="query",
        type="string",
        description="Search query",
        required=True
    )
    assert param.name == "query"
    assert param.type == "string"
    assert param.required is True


def test_tool_result_creation():
    """Test ToolResult can be created."""
    result = ToolResult(
        tool_name="test_tool",
        success=True,
        data=["result1", "result2"],
        execution_time=0.5,
        metadata={"count": 2}
    )
    assert result.tool_name == "test_tool"
    assert result.success is True
    assert len(result.data) == 2
    assert result.execution_time == 0.5


def test_tool_to_anthropic_format():
    """Test tool conversion to Anthropic format."""
    class TestTool(Tool):
        @property
        def name(self):
            return "test_tool"
        
        @property
        def description(self):
            return "Test tool description"
        
        @property
        def parameters(self):
            return [
                ToolParameter(name="query", type="string", description="Query", required=True),
                ToolParameter(name="top_k", type="integer", description="Top K", required=False, default=5)
            ]
        
        def execute(self, **kwargs):
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=[],
                execution_time=0.0
            )
    
    tool = TestTool()
    anthropic_format = tool.to_anthropic_tool()
    
    assert anthropic_format["name"] == "test_tool"
    assert "description" in anthropic_format
    assert "input_schema" in anthropic_format
    assert "query" in anthropic_format["input_schema"]["properties"]
    assert "query" in anthropic_format["input_schema"]["required"]
    assert "top_k" not in anthropic_format["input_schema"]["required"]


# Test SemanticSearchTool

def test_semantic_search_tool_definition(mock_chunk_repository, mock_embedding_service):
    """Test semantic search tool has correct definition."""
    tool = SemanticSearchTool(mock_chunk_repository, mock_embedding_service)
    
    assert tool.name == "semantic_search"
    assert len(tool.description) > 0
    assert len(tool.parameters) == 3
    assert tool.parameters[0].name == "query"
    assert tool.parameters[0].required is True


def test_semantic_search_tool_execute(mock_chunk_repository, mock_embedding_service):
    """Test semantic search tool execution."""
    tool = SemanticSearchTool(mock_chunk_repository, mock_embedding_service)
    
    result = tool.execute(query="test query", top_k=5)
    
    assert result.success is True
    assert result.tool_name == "semantic_search"
    assert len(result.data) == 2
    assert result.execution_time > 0
    assert result.metadata["query"] == "test query"
    
    mock_embedding_service.embed_text.assert_called_once_with("test query")
    mock_chunk_repository.search_similar.assert_called_once()


def test_semantic_search_tool_error_handling(mock_chunk_repository, mock_embedding_service):
    """Test semantic search tool handles errors."""
    mock_embedding_service.embed_text.side_effect = Exception("Embedding failed")
    tool = SemanticSearchTool(mock_chunk_repository, mock_embedding_service)
    
    result = tool.execute(query="test query")
    
    assert result.success is False
    assert result.error == "Embedding failed"
    assert result.data == []


# Test DocumentReaderTool

def test_document_reader_tool(mock_document_repository, mock_chunk_repository):
    """Test document reader tool."""
    tool = DocumentReaderTool(mock_document_repository, mock_chunk_repository)
    
    doc_id = str(uuid4())
    result = tool.execute(document_id=doc_id)
    
    assert result.success is True
    assert result.data is not None
    assert "filename" in result.data
    assert "chunks" in result.data
    assert len(result.data["chunks"]) == 2


def test_document_reader_tool_not_found(mock_document_repository, mock_chunk_repository):
    """Test document reader tool when document not found."""
    mock_document_repository.get_by_id.return_value = None
    tool = DocumentReaderTool(mock_document_repository, mock_chunk_repository)
    
    doc_id = str(uuid4())
    result = tool.execute(document_id=doc_id)
    
    assert result.success is False
    assert "not found" in result.error


def test_document_reader_tool_invalid_id(mock_document_repository, mock_chunk_repository):
    """Test document reader tool with invalid ID."""
    tool = DocumentReaderTool(mock_document_repository, mock_chunk_repository)
    
    result = tool.execute(document_id="invalid-uuid")
    
    assert result.success is False
    assert "Invalid document ID" in result.error


# Test MetadataSearchTool

def test_metadata_search_tool(mock_chunk_repository, mock_embedding_service):
    """Test metadata search tool."""
    tool = MetadataSearchTool(mock_chunk_repository, mock_embedding_service)
    
    filters = {"author": "John", "year": "2024"}
    result = tool.execute(query="test", metadata_filter=filters, top_k=5)
    
    assert result.success is True
    assert len(result.data) == 1
    assert result.metadata["metadata_filter"] == filters


# Test HybridSearchTool

def test_hybrid_search_tool(mock_chunk_repository, mock_embedding_service):
    """Test hybrid search tool."""
    tool = HybridSearchTool(mock_chunk_repository, mock_embedding_service)
    
    result = tool.execute(query="test query", top_k=5, semantic_weight=0.7)
    
    assert result.success is True
    assert result.tool_name == "hybrid_search"
    assert len(result.data) >= 0
    assert result.metadata["semantic_weight"] == 0.7
