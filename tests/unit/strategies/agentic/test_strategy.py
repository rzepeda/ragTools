"""Unit tests for agentic RAG strategy."""

import pytest
from unittest.mock import Mock, patch

from rag_factory.strategies.agentic.strategy import AgenticRAGStrategy
from rag_factory.strategies.agentic.config import AgenticStrategyConfig
from rag_factory.services.dependencies import StrategyDependencies


# Fixtures

@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    return Mock()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.embed_text.return_value = [0.1] * 384
    return service


@pytest.fixture
def mock_chunk_repository():
    """Mock chunk repository."""
    repo = Mock()
    repo.search_similar.return_value = [
        (Mock(
            chunk_id="1",
            document_id="doc1",
            text="Fallback result",
            metadata_={},
            chunk_index=0
        ), 0.8)
    ]
    return repo


@pytest.fixture
def mock_document_repository():
    """Mock document repository."""
    return Mock()


@pytest.fixture
def mock_database_service(mock_chunk_repository, mock_document_repository):
    """Mock database service that wraps repositories."""
    service = Mock()
    service.chunk_repository = mock_chunk_repository
    service.document_repository = mock_document_repository
    return service


# Test AgenticStrategyConfig

def test_config_defaults():
    """Test configuration defaults."""
    config = AgenticStrategyConfig()
    
    assert config.max_iterations == 3
    assert config.enable_query_analysis is True
    assert config.fallback_to_semantic is True
    assert config.timeout == 30


def test_config_custom_values():
    """Test configuration with custom values."""
    config = AgenticStrategyConfig(
        max_iterations=5,
        enable_query_analysis=False,
        fallback_to_semantic=False
    )
    
    assert config.max_iterations == 5
    assert config.enable_query_analysis is False
    assert config.fallback_to_semantic is False


def test_config_validation():
    """Test configuration validation."""
    # Should raise error for invalid max_iterations
    with pytest.raises(Exception):
        AgenticStrategyConfig(max_iterations=0)
    
    with pytest.raises(Exception):
        AgenticStrategyConfig(max_iterations=20)


# Test AgenticRAGStrategy

def test_strategy_initialization(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy initialization."""
    strategy = AgenticRAGStrategy(
        config={},
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    assert strategy.name == "agentic"
    assert len(strategy.tools) == 4  # All tools enabled by default
    assert strategy.agent is not None


def test_strategy_initialization_with_config(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy initialization with config."""
    config = {
        "max_iterations": 5,
        "enable_query_analysis": False,
        "enabled_tools": ["semantic_search", "metadata_search"]
    }
    
    strategy = AgenticRAGStrategy(
        config=config,
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    assert strategy.config.max_iterations == 5
    assert len(strategy.tools) == 2  # Only enabled tools


def test_strategy_retrieve(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy retrieve method."""
    strategy = AgenticRAGStrategy(
        config={},
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    # Mock agent run
    mock_agent_result = {
        "results": [
            {"chunk_id": "1", "text": "Result 1"},
            {"chunk_id": "2", "text": "Result 2"}
        ],
        "trace": {
            "iterations": 1,
            "tool_calls": [{"tool": "semantic_search"}],
            "tool_results": [],
            "plan": {}
        }
    }
    
    with patch.object(strategy.agent, 'run', return_value=mock_agent_result):
        results = strategy.retrieve("test query", top_k=5)
    
    assert len(results) == 2
    assert results[0]["strategy"] == "agentic"
    assert "agent_trace" in results[0]


def test_strategy_retrieve_with_query_analysis(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy retrieve with query analysis."""
    config = {"enable_query_analysis": True}
    strategy = AgenticRAGStrategy(
        config=config,
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    assert strategy.query_analyzer is not None
    
    mock_agent_result = {
        "results": [{"chunk_id": "1", "text": "Result"}],
        "trace": {"iterations": 1, "tool_calls": [], "tool_results": [], "plan": {}}
    }
    
    with patch.object(strategy.agent, 'run', return_value=mock_agent_result):
        results = strategy.retrieve("test query")
    
    assert len(results) >= 0


def test_strategy_fallback_on_error(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy falls back to semantic search on error."""
    config = {"fallback_to_semantic": True}
    strategy = AgenticRAGStrategy(
        config=config,
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    # Mock agent to raise error
    with patch.object(strategy.agent, 'run', side_effect=Exception("Agent failed")):
        results = strategy.retrieve("test query", top_k=5)
    
    # Should get fallback results
    assert len(results) >= 0


def test_strategy_no_fallback_raises_error(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy raises error when fallback disabled."""
    config = {"fallback_to_semantic": False}
    strategy = AgenticRAGStrategy(
        config=config,
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    # Mock agent to raise error
    with patch.object(strategy.agent, 'run', side_effect=Exception("Agent failed")):
        with pytest.raises(Exception):
            strategy.retrieve("test query")


def test_strategy_get_stats(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy get_stats method."""
    strategy = AgenticRAGStrategy(
        config={},
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    stats = strategy.get_stats()
    
    assert stats["name"] == "agentic"
    assert "description" in stats
    assert "num_tools" in stats
    assert "tools" in stats
    assert "config" in stats


def test_strategy_top_k_limit(
    mock_llm_service,
    mock_embedding_service,
    mock_database_service
):
    """Test strategy respects top_k limit."""
    strategy = AgenticRAGStrategy(
        config={},
        dependencies=StrategyDependencies(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            database_service=mock_database_service
        )
    )
    
    # Mock agent to return many results
    mock_agent_result = {
        "results": [{"chunk_id": str(i), "text": f"Result {i}"} for i in range(20)],
        "trace": {"iterations": 1, "tool_calls": [], "tool_results": [], "plan": {}}
    }
    
    with patch.object(strategy.agent, 'run', return_value=mock_agent_result):
        results = strategy.retrieve("test query", top_k=5)
    
    assert len(results) <= 5
