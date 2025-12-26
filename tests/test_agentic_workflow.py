import pytest
import asyncio
import pytest_asyncio
import uuid
from unittest.mock import patch
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.services.database.postgres import PostgresqlDatabaseService
from sqlalchemy.orm import Session
from sqlalchemy import create_engine


@pytest.fixture(scope="module")
def service_registry():
    """Fixture for a ServiceRegistry instance."""
    return ServiceRegistry("config/services.yaml")

@pytest.fixture(scope="module")
def strategy_pair_manager(service_registry):
    """Fixture for a StrategyPairManager instance."""
    return StrategyPairManager(service_registry)

import os
from pathlib import Path
from dotenv import load_dotenv

@pytest.fixture(scope="function")
def db_session():
    """Fixture to provide a function-scoped, transactional database session."""
    # Load .env file from project root
    # This assumes the test is run from the project root or similar context
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    print(f"DEBUG (db_session): Attempting to load .env from: {env_file}")
    if env_file.exists():
        print(f"DEBUG (db_session): .env file found. Loading...")
        load_dotenv(env_file)
        print(f"DEBUG (db_session): .env loaded.")
    else:
        print(f"DEBUG (db_session): .env file NOT found at: {env_file}")

    # Prioritize TEST_DATABASE_URL, then DATABASE_URL
    db_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    print(f"DEBUG (db_session): Resolved DB_URL: {db_url}")
    if not db_url:
        pytest.fail("DATABASE_URL or TEST_DATABASE_URL environment variable not set. Please create a .env file with one of these variables.")
    
    # Create an engine for the test session
    engine = create_engine(
        db_url,
        isolation_level="SERIALIZABLE" # Ensure strong isolation
    )
    
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    # Run migrations within the transaction
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    yield session

    session.close()
    transaction.rollback()
    connection.close()
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_service_for_setup(db_session):
    """Fixture to provide a function-scoped PostgresqlDatabaseService with injected session."""
    # Create a new instance of PostgresqlDatabaseService
    # Its __init__ will use default engine creation, but we will override it
    db_service_instance = PostgresqlDatabaseService()
    
    # Inject the test-managed engine (from db_session's connection)
    db_service_instance.engine = db_session.bind
    
    # Also ensure its internal session for repositories is set to our db_session
    # This might be tricky if _session is lazily created.
    # We can explicitly set it here to ensure it uses the test transaction.
    db_service_instance._session = db_session
    
    yield db_service_instance
    
    # Cleanup for the injected service
    # The session and connection are closed by db_session fixture
    db_service_instance._session = None
    db_service_instance._sync_engine = None

@pytest.fixture(scope="function")
def test_service_registry(test_db_service_for_setup):
    """Fixture for a ServiceRegistry instance using the test-scoped db_service."""
    registry = ServiceRegistry("config/services.yaml")
    # Manually inject our test-controlled db_service into the registry
    registry._instances["db_main"] = test_db_service_for_setup
    return registry

@pytest.fixture(scope="function")
def test_strategy_pair_manager(test_service_registry):
    """Fixture for a StrategyPairManager instance using the test-scoped service_registry."""
    return StrategyPairManager(test_service_registry)

@pytest_asyncio.fixture(scope="function")
async def setup_agentic_strategy(test_strategy_pair_manager):
    """Fixture to set up the agentic-rag-pair strategy and index data."""
    strategy_name = "agentic-rag-pair"
    indexing_strategy, retrieval_strategy = test_strategy_pair_manager.load_pair(strategy_name)

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

    db_service = indexing_strategy.deps.database_service
    from rag_factory.core.indexing_interface import IndexingContext
    indexing_context = IndexingContext(database_service=db_service, config={})
    
    # The process method now uses repositories, so its operations are part of the session
    await indexing_strategy.process(doc_to_index, indexing_context)
    
    # No explicit commit needed here, as the db_session fixture will handle the transaction.
    # The yield will effectively make the data visible within the current transaction.

    yield retrieval_strategy, test_data

    # Cleanup is handled by the db_session fixture's rollback.
    # No explicit cleanup of the document is needed here.

@pytest.mark.asyncio
@patch('rag_factory.strategies.agentic.agent.SimpleAgent._select_workflow')
async def test_agentic_workflow_plan_1(mock_select_workflow, setup_agentic_strategy):
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
async def test_agentic_workflow_plan_6(mock_select_workflow, setup_agentic_strategy):
    retrieval_strategy, test_data = setup_agentic_strategy
    mock_select_workflow.return_value = {"plan": 6, "reasoning": "Test-forced plan 6"}
    query = f"read document {test_data['id']}"
    retrieved_chunks = await retrieval_strategy.retrieve(query=query, context={"top_k": 5})
    assert isinstance(retrieved_chunks, list)
    assert len(retrieved_chunks) > 0
    print(f"Successfully tested workflow 6. Found {len(retrieved_chunks)} chunks.")