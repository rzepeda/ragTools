import sys
import importlib.util
from unittest.mock import MagicMock
import pytest

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This ensures .env is loaded before any tests run

# Mock numpy and other dependencies GLOBALLY before any test collection
try:
    import numpy
except ImportError:
    sys.modules["numpy"] = MagicMock()

# Mock the services package and submodules that cause issues
# sys.modules["rag_factory.services"] = MagicMock()
# sys.modules["rag_factory.services.onnx"] = MagicMock()
# sys.modules["rag_factory.services.onnx.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers.onnx_local"] = MagicMock()
# sys.modules["rag_factory.services.embedding.service"] = MagicMock()
# sys.modules["rag_factory.services.api"] = MagicMock()
# sys.modules["rag_factory.services.database"] = MagicMock()
# sys.modules["rag_factory.services.local"] = MagicMock()

# Manually load the modules we actually need
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Mock rag_factory package to prevent init
# sys.modules["rag_factory"] = MagicMock()
# sys.modules["rag_factory.strategies"] = MagicMock()

# Load modules in dependency order to avoid cycles and init triggers
load_module("rag_factory.core.capabilities", "/mnt/MCPProyects/ragTools/rag_factory/core/capabilities.py")
load_module("rag_factory.services.interfaces", "/mnt/MCPProyects/ragTools/rag_factory/services/interfaces.py")
load_module("rag_factory.services.dependencies", "/mnt/MCPProyects/ragTools/rag_factory/services/dependencies.py")
load_module("rag_factory.services.consistency", "/mnt/MCPProyects/ragTools/rag_factory/services/consistency.py")
load_module("rag_factory.core.indexing_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/indexing_interface.py")
load_module("rag_factory.core.retrieval_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/retrieval_interface.py")
load_module("rag_factory.core.pipeline", "/mnt/MCPProyects/ragTools/rag_factory/core/pipeline.py")
load_module("rag_factory.strategies.base", "/mnt/MCPProyects/ragTools/rag_factory/strategies/base.py")
load_module("rag_factory.exceptions", "/mnt/MCPProyects/ragTools/rag_factory/exceptions.py")
load_module("rag_factory.factory", "/mnt/MCPProyects/ragTools/rag_factory/factory.py")
load_module("rag_factory.strategies.indexing", "/mnt/MCPProyects/ragTools/rag_factory/strategies/indexing/__init__.py")

# Load CLI modules for CLI tests
load_module("rag_factory.cli", "/mnt/MCPProyects/ragTools/rag_factory/cli/__init__.py")
load_module("rag_factory.cli.formatters", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/__init__.py")
load_module("rag_factory.cli.formatters.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/validation.py")
load_module("rag_factory.cli.utils", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/__init__.py")
load_module("rag_factory.cli.utils.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/validation.py")
load_module("rag_factory.cli.commands", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/__init__.py")
load_module("rag_factory.cli.commands.validate_pipeline", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/validate_pipeline.py")
load_module("rag_factory.cli.main", "/mnt/MCPProyects/ragTools/rag_factory/cli/main.py")


# =============================================================================
# Database Fixtures
# =============================================================================

import os
import asyncio
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from urllib.parse import urlparse

from rag_factory.database.models import Base
from rag_factory.database.config import DatabaseConfig
from rag_factory.database.connection import DatabaseConnection
from rag_factory.services.database.postgres import PostgresqlDatabaseService


# =============================================================================
# Database URL Fixture
# =============================================================================

@pytest.fixture(scope="session")
def test_db_url() -> str:
    """
    Get test database URL from environment.
    
    Returns:
        Test database URL
        
    Raises:
        pytest.skip: If DB_TEST_DATABASE_URL not set
    """
    url = os.getenv("DB_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "DB_TEST_DATABASE_URL not set. "
            "Set this environment variable to run database tests."
        )
    return url


# =============================================================================
# SQLAlchemy Engine Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def db_engine(test_db_url: str):
    """
    Create database engine for testing.
    
    Creates all tables before tests and drops them after.
    Uses session scope for efficiency.
    
    Args:
        test_db_url: Test database URL from environment
        
    Yields:
        SQLAlchemy Engine instance
    """
    # Create engine
    engine = create_engine(
        test_db_url,
        poolclass=StaticPool,  # Use static pool for tests
        echo=False,  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Drop all tables and dispose engine
    # Use CASCADE to handle dependent views created by migrations
    with engine.connect() as conn:
        # Drop views first if they exist
        conn.execute(text("DROP VIEW IF EXISTS chunk_hierarchy_validation CASCADE"))
        conn.commit()
    
    Base.metadata.drop_all(engine)
    engine.dispose()


# =============================================================================
# Database Connection Fixtures (Sync)
# =============================================================================

@pytest.fixture(scope="function")
def db_connection(test_db_config: DatabaseConfig) -> Generator[DatabaseConnection, None, None]:
    """
    Provide database connection with transaction rollback.
    
    Each test runs in an isolated transaction that is rolled back
    after the test completes. This ensures no test data persists.
    
    Args:
        test_db_config: Database configuration from test_db_config fixture
        
    Yields:
        DatabaseConnection instance with transaction isolation
        
    Example:
        ```python
        def test_insert_document(db_connection):
            with db_connection.get_session() as session:
                doc = Document(filename="test.txt")
                session.add(doc)
                session.commit()
                
                # Query to verify
                result = session.query(Document).first()
                assert result.filename == "test.txt"
            # Transaction will be rolled back after test
        ```
    """
    # Create DatabaseConnection instance
    db = DatabaseConnection(test_db_config)
    
    # Create tables
    db.create_tables()
    
    yield db
    
    # Cleanup: drop tables and close connection
    try:
        db.drop_tables()
    except Exception:
        pass  # Tables might already be dropped
    db.close()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    Provide raw SQLAlchemy Session with transaction rollback.
    
    This fixture is for tests that need direct Session access (like model tests).
    Each test runs in an isolated transaction that is rolled back.
    
    Args:
        db_engine: SQLAlchemy engine from session fixture
        
    Yields:
        SQLAlchemy Session with transaction isolation
        
    Example:
        ```python
        def test_create_document(db_session):
            doc = Document(filename="test.txt")
            db_session.add(doc)
            db_session.flush()
            
            result = db_session.query(Document).first()
            assert result.filename == "test.txt"
            # Transaction automatically rolled back after test
        ```
    """
    # Create connection
    connection = db_engine.connect()
    
    # Begin transaction
    transaction = connection.begin()
    
    # Create session bound to connection
    SessionMaker = sessionmaker(bind=connection)
    session = SessionMaker()
    
    # Yield session to test
    yield session
    
    # Cleanup: close session, rollback transaction, close connection
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def test_db_config(test_db_url: str) -> DatabaseConfig:
    """
    Provide database configuration for tests.
    
    Args:
        test_db_url: Test database URL from environment
        
    Returns:
        DatabaseConfig instance
        
    Example:
        ```python
        def test_database_config(test_db_config):
            assert test_db_config.database_url.startswith("postgresql://")
        ```
    """
    return DatabaseConfig(database_url=test_db_url)


# =============================================================================
# Database Service Fixtures (Async)
# =============================================================================

# Import pytest_asyncio for async fixtures
try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

@pytest_asyncio.fixture(scope="function") if PYTEST_ASYNCIO_AVAILABLE else pytest.fixture(scope="function")
async def db_service(test_db_url: str) -> AsyncGenerator[PostgresqlDatabaseService, None]:
    """
    Provide async database service for integration tests.
    
    Creates a PostgresqlDatabaseService instance configured for testing.
    Automatically closes the service after the test.
    
    Args:
        test_db_url: Test database URL from environment
        
    Yields:
        PostgresqlDatabaseService instance
        
    Example:
        ```python
        @pytest.mark.asyncio
        async def test_store_chunks(db_service):
            chunks = [
                {"id": "1", "text": "test", "embedding": [0.1, 0.2]}
            ]
            await db_service.store_chunks(chunks)
            
            results = await db_service.search_chunks([0.1, 0.2], top_k=1)
            assert len(results) == 1
        ```
    """
    # Parse URL for connection parameters
    parsed = urlparse(test_db_url)
    
    # Create service (pool is created lazily on first use)
    service = PostgresqlDatabaseService(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username or 'postgres',
        password=parsed.password or ''
    )
    
    yield service
    
    # Cleanup: close all connections
    await service.close()


@pytest.fixture(scope="function")
def db_config(test_db_url: str) -> DatabaseConfig:
    """
    Provide database configuration for tests.
    
    Args:
        test_db_url: Test database URL from environment
        
    Returns:
        DatabaseConfig instance
        
    Example:
        ```python
        def test_database_config(db_config):
            assert db_config.database_url.startswith("postgresql://")
        ```
    """
    return DatabaseConfig(database_url=test_db_url)


# =============================================================================
# Test Database Utilities
# =============================================================================

@pytest.fixture(scope="function")
def clean_database(db_connection):
    """
    Ensure database is clean before and after test.
    
    Deletes all data from all tables before and after the test.
    Use this for tests that need a completely empty database.
    
    Args:
        db_connection: Database connection fixture
        
    Yields:
        Database connection with clean state
        
    Example:
        ```python
        def test_with_clean_db(clean_database):
            # Database is guaranteed to be empty
            count = clean_database.query(Document).count()
            assert count == 0
        ```
    """
    # Clean before test
    _clean_all_tables(db_connection)
    
    yield db_connection
    
    # Clean after test
    _clean_all_tables(db_connection)


def _clean_all_tables(session: Session) -> None:
    """
    Delete all data from all tables.
    
    Args:
        session: SQLAlchemy session
    """
    # Get all table names in reverse order (to handle foreign keys)
    tables = reversed(Base.metadata.sorted_tables)
    
    for table in tables:
        session.execute(table.delete())
    
    session.commit()


# =============================================================================
# LLM Service Fixtures
# =============================================================================

@pytest.fixture
def llm_service_from_env():
    """
    Create LLM service using .env configuration (LM Studio by default).
    
    This fixture uses the LLM configuration from environment variables,
    which defaults to LM Studio for local development and testing.
    
    Returns:
        LLMService instance configured from environment
        
    Example:
        ```python
        def test_with_lm_studio(llm_service_from_env):
            response = llm_service_from_env.complete([
                Message(role=MessageRole.USER, content="Hello")
            ])
            assert response.content
        ```
    """
    from rag_factory.services.llm.service import LLMService
    from rag_factory.services.llm.config import LLMServiceConfig
    
    config = LLMServiceConfig(
        provider="openai",  # LM Studio uses OpenAI-compatible API
        model=os.getenv("OPENAI_MODEL", "default-model"),
        provider_config={
            "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
            "base_url": os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        }
    )
    return LLMService(config)


@pytest.fixture
def mock_llm_service():
    """
    Create fully mocked LLM service for fast unit tests.
    
    This fixture provides a mock LLM service that returns predictable
    responses without making actual API calls. Use for tests that need
    LLM functionality but don't require real model inference.
    
    Returns:
        Mock LLMService instance
        
    Example:
        ```python
        def test_with_mock_llm(mock_llm_service):
            response = mock_llm_service.complete([...])
            assert response.text == "Mocked LLM response"
        ```
    """
    from unittest.mock import Mock, AsyncMock
    
    service = Mock()
    
    # Create mock response
    response = Mock()
    response.text = "Mocked LLM response"
    response.content = "Mocked LLM response"
    response.prompt_tokens = 10
    response.completion_tokens = 20
    response.total_tokens = 30
    response.cost = 0.0
    response.latency = 0.1
    
    # Setup mock methods
    service.complete = Mock(return_value=response)
    service.agenerate = AsyncMock(return_value=response)
    service.count_tokens = Mock(return_value=10)
    service.get_stats = Mock(return_value={
        "total_requests": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 20,
        "total_cost": 0.0,
        "total_latency": 0.1,
        "average_latency": 0.1,
        "model": "mock-model",
        "provider": "mock"
    })
    service.estimate_cost = Mock(return_value=0.0)
    
    return service


@pytest.fixture
def cloud_llm_service():
    """
    Create LLM service using cloud providers (for benchmarking).
    
    This fixture uses cloud API keys (OpenAI or Anthropic) and is intended
    for benchmarking tests that compare performance across providers.
    Tests using this fixture will be skipped if cloud API keys are not available.
    
    Returns:
        LLMService instance configured for cloud provider
        
    Raises:
        pytest.skip: If no cloud API keys are available
        
    Example:
        ```python
        @pytest.mark.benchmark
        @pytest.mark.requires_cloud_api
        def test_performance(cloud_llm_service):
            # Benchmarking test using cloud provider
            pass
        ```
    """
    from rag_factory.services.llm.service import LLMService
    from rag_factory.services.llm.config import LLMServiceConfig
    
    # Try OpenAI first (use different env var to distinguish from LM Studio)
    if os.getenv("OPENAI_CLOUD_API_KEY"):
        config = LLMServiceConfig(
            provider="openai",
            model=os.getenv("OPENAI_CLOUD_MODEL", "gpt-3.5-turbo"),
            provider_config={"api_key": os.getenv("OPENAI_CLOUD_API_KEY")}
        )
    # Fall back to Anthropic
    elif os.getenv("ANTHROPIC_API_KEY"):
        config = LLMServiceConfig(
            provider="anthropic",
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")}
        )
    else:
        pytest.skip("No cloud API keys available for benchmarking")
    
    return LLMService(config)


# =============================================================================
# Centralized Mock Fixtures
# =============================================================================

# Import centralized mock builders
from tests.mocks import (
    create_mock_embedding_service,
    create_mock_database_service,
    create_mock_llm_service,
    create_mock_neo4j_service,
    create_mock_registry_with_services,
    create_mock_migration_validator,
    create_mock_onnx_environment,
)


@pytest.fixture
def mock_embedding_service():
    """Provide standard mock embedding service (384 dimensions).
    
    Returns:
        Mock embedding service with standard configuration
        
    Example:
        ```python
        def test_with_embedding(mock_embedding_service):
            result = await mock_embedding_service.embed("test")
            assert len(result) == 384
        ```
    """
    return create_mock_embedding_service(dimension=384)


@pytest.fixture
def mock_embedding_service_768():
    """Provide mock embedding service with 768 dimensions.
    
    Returns:
        Mock embedding service (768-dim)
    """
    return create_mock_embedding_service(dimension=768)


@pytest.fixture
def mock_database_service():
    """Provide standard mock database service.
    
    Returns:
        Mock database service with CRUD operations
        
    Example:
        ```python
        async def test_with_db(mock_database_service):
            await mock_database_service.store_chunks(chunks)
            results = await mock_database_service.search_chunks(embedding)
        ```
    """
    return create_mock_database_service()


@pytest.fixture
def mock_llm_service():
    """Provide standard mock LLM service.
    
    Returns:
        Mock LLM service
        
    Example:
        ```python
        async def test_with_llm(mock_llm_service):
            response = await mock_llm_service.agenerate("prompt")
            assert response == "Mock LLM response"
        ```
    """
    return create_mock_llm_service()


@pytest.fixture
def mock_neo4j_service():
    """Provide standard mock Neo4j service.
    
    Returns:
        Mock Neo4j graph database service
    """
    return create_mock_neo4j_service()


@pytest.fixture
def mock_registry_with_services():
    """Provide mock registry with embedding and database services.
    
    This is the most commonly used fixture for integration tests.
    Includes:
    - Mock embedding service (384 dimensions)
    - Mock database service
    - Mock migration validator
    
    Returns:
        Mock service registry
        
    Example:
        ```python
        def test_strategy_pair(mock_registry_with_services):
            manager = StrategyPairManager(
                service_registry=mock_registry_with_services,
                config_dir=str(config_dir)
            )
            indexing, retrieval = manager.load_pair("semantic-local-pair")
        ```
    """
    return create_mock_registry_with_services(
        include_embedding=True,
        include_database=True,
        include_llm=False,
        include_neo4j=False
    )


@pytest.fixture
def mock_registry_with_llm_services():
    """Provide mock registry with LLM, embedding, and database services.
    
    Use this for tests that require LLM functionality.
    
    Returns:
        Mock service registry with LLM support
    """
    return create_mock_registry_with_services(
        include_embedding=True,
        include_database=True,
        include_llm=True,
        include_neo4j=False
    )


@pytest.fixture
def mock_registry_with_graph_services():
    """Provide mock registry with all services including Neo4j.
    
    Use this for knowledge graph tests.
    
    Returns:
        Mock service registry with graph database support
    """
    return create_mock_registry_with_services(
        include_embedding=True,
        include_database=True,
        include_llm=True,
        include_neo4j=True
    )


@pytest.fixture
def mock_registry_with_reranker_services():
    """Provide mock registry with reranker service support.
    
    Use this for reranking tests.
    
    Returns:
        Mock service registry with reranker support
    """
    return create_mock_registry_with_services(
        include_embedding=True,
        include_database=True,
        include_llm=False,
        include_reranker=True
    )


@pytest.fixture
def mock_migration_validator():
    """Provide mock migration validator (always valid).
    
    Returns:
        Mock migration validator
    """
    return create_mock_migration_validator(is_valid=True)


@pytest.fixture
def mock_onnx_env():
    """Provide ONNX environment context manager.
    
    Returns:
        Context manager for ONNX mocking
        
    Example:
        ```python
        def test_onnx_provider(mock_onnx_env):
            with mock_onnx_env:
                provider = ONNXLocalProvider(config)
                embeddings = provider.get_embeddings(["text"])
        ```
    """
    return create_mock_onnx_environment


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers",
        "database: mark test as requiring database connection"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as benchmarking test (requires cloud API keys)"
    )
    config.addinivalue_line(
        "markers",
        "requires_cloud_api: mark test as requiring cloud API keys (OpenAI/Anthropic)"
    )
    config.addinivalue_line(
        "markers",
        "requires_llm: mark test as requiring LLM service"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip tests based on environment configuration.
    
    Args:
        config: Pytest configuration
        items: Collected test items
    """
    skip_db = pytest.mark.skip(reason="DB_TEST_DATABASE_URL not set")
    skip_cloud_api = pytest.mark.skip(reason="Cloud API keys not available for benchmarking")
    
    # Check if database URL is set
    db_url_set = bool(os.getenv("DB_TEST_DATABASE_URL"))
    
    # Check if cloud API keys are set
    cloud_api_available = bool(
        os.getenv("OPENAI_CLOUD_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )
    
    for item in items:
        # Skip database tests if not configured
        if not db_url_set and "database" in item.keywords:
            item.add_marker(skip_db)
        
        # Skip benchmarking/cloud API tests if not configured
        if not cloud_api_available and "requires_cloud_api" in item.keywords:
            item.add_marker(skip_cloud_api)


