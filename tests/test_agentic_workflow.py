import pytest
import asyncio
import pytest_asyncio
import uuid
from unittest.mock import patch

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry

@pytest.fixture(scope="module")
def service_registry():
    """Fixture for a ServiceRegistry instance."""
    return ServiceRegistry("config/services.yaml")

@pytest.fixture(scope="module")
def strategy_pair_manager(service_registry):
    """Fixture for a StrategyPairManager instance."""
    return StrategyPairManager(service_registry)

@pytest_asyncio.fixture(scope="module")
async def setup_agentic_strategy(strategy_pair_manager):
    """Fixture to set up the agentic-rag-pair strategy and index data."""
    strategy_name = "agentic-rag-pair"
    indexing_strategy, retrieval_strategy = strategy_pair_manager.load_pair(strategy_name)

    test_data = {
        "id": str(uuid.uuid4()),
        "text": "The quick brown fox jumps over the lazy dog. This document is about animals.",
        "query": "quick brown fox",
        "metadata": {"author": "animal_lover", "year": "2023"}
    }
    doc_to_index = [{
        "id": test_data["id"],
        "text": test_data["text"],
        "metadata": test_data["metadata"]
    }]

    from rag_factory.core.indexing_interface import IndexingContext
    db_service = indexing_strategy.deps.database_service
    indexing_context = IndexingContext(database_service=db_service, config={})
    await indexing_strategy.process(doc_to_index, indexing_context)

    return retrieval_strategy, test_data

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_1_passes(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 1, "reasoning": "Test-forced plan 1"}
    retrieved_chunks = await retrieval_strategy.retrieve(query=test_data["query"], context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 1. Found {len(retrieved_chunks)} chunks.")

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_2_fails(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 2, "reasoning": "Test-forced plan 2"}
    retrieved_chunks = await retrieval_strategy.retrieve(query="documents by author animal_lover from 2023", context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 2. Found {len(retrieved_chunks)} chunks.")

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_3_fails(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 3, "reasoning": "Test-forced plan 3"}
    retrieved_chunks = await retrieval_strategy.retrieve(query=test_data["query"], context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 3. Found {len(retrieved_chunks)} chunks.")

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_4_fails(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 4, "reasoning": "Test-forced plan 4"}
    retrieved_chunks = await retrieval_strategy.retrieve(query=test_data["query"], context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 4. Found {len(retrieved_chunks)} chunks.")

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_5_fails(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 5, "reasoning": "Test-forced plan 5"}
    retrieved_chunks = await retrieval_strategy.retrieve(query=test_data["query"], context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 5. Found {len(retrieved_chunks)} chunks.")

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_6_fails(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 6, "reasoning": "Test-forced plan 6"}
    query = f"read document {test_data['id']}"
    retrieved_chunks = await retrieval_strategy.retrieve(query=query, context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 6. Found {len(retrieved_chunks)} chunks.")