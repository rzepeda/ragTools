import os
import pytest
from typing import Generator
import psycopg2
from neo4j import GraphDatabase


# Custom markers for conditional test execution
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_db: mark test as requiring database connection")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM service")
    config.addinivalue_line("markers", "requires_openai: mark test as requiring OpenAI API key")
    config.addinivalue_line("markers", "requires_cohere: mark test as requiring Cohere API key")
    config.addinivalue_line("markers", "requires_ollama: mark test as requiring Ollama server")
    config.addinivalue_line("markers", "requires_neo4j: mark test as requiring Neo4j database")
    config.addinivalue_line("markers", "requires_onnx: mark test as requiring ONNX models")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on environment configuration"""
    
    run_db_tests = os.getenv("RUN_DB_TESTS", "false").lower() == "true"
    run_llm_tests = os.getenv("RUN_LLM_TESTS", "false").lower() == "true"
    
    skip_db = pytest.mark.skip(reason="Database not configured (RUN_DB_TESTS=false)")
    skip_llm = pytest.mark.skip(reason="LLM service not configured (RUN_LLM_TESTS=false)")
    skip_openai = pytest.mark.skip(reason="OpenAI API key not configured")
    skip_cohere = pytest.mark.skip(reason="Cohere API key not configured")
    skip_ollama = pytest.mark.skip(reason="Ollama server not configured")
    skip_neo4j = pytest.mark.skip(reason="Neo4j not configured")
    
    for item in items:
        # Skip database tests if not configured
        if "requires_db" in item.keywords and not run_db_tests:
            item.add_marker(skip_db)
        
        # Skip LLM tests if not configured
        if "requires_llm" in item.keywords and not run_llm_tests:
            item.add_marker(skip_llm)
        
        # Skip specific provider tests based on API keys
        if "requires_openai" in item.keywords and not os.getenv("OPENAI_API_KEY"):
            item.add_marker(skip_openai)
        
        if "requires_cohere" in item.keywords and not os.getenv("COHERE_API_KEY"):
            item.add_marker(skip_cohere)
        
        if "requires_ollama" in item.keywords and not is_ollama_available():
            item.add_marker(skip_ollama)
        
        if "requires_neo4j" in item.keywords and not is_neo4j_available():
            item.add_marker(skip_neo4j)


def is_ollama_available() -> bool:
    """Check if Ollama server is available"""
    import requests
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def is_neo4j_available() -> bool:
    """Check if Neo4j is available"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "rag_password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except:
        return False


# Database fixtures
@pytest.fixture(scope="session")
def db_connection():
    """PostgreSQL database connection fixture"""
    if not os.getenv("RUN_DB_TESTS", "false").lower() == "true":
        pytest.skip("Database tests not enabled")
    
    database_url = os.getenv("DATABASE_TEST_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not configured")
    
    # Parse connection string
    # Format: postgresql://user:password@host:port/database
    import re
    match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
    if not match:
        pytest.skip("Invalid DATABASE_URL format")
    
    user, password, host, port, dbname = match.groups()
    
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    conn.autocommit = False
    
    yield conn
    
    conn.rollback()
    conn.close()


@pytest.fixture(scope="function")
def db_session(db_connection):
    """Database session with automatic rollback"""
    cursor = db_connection.cursor()
    yield cursor
    db_connection.rollback()
    cursor.close()


@pytest.fixture(scope="session")
def neo4j_driver():
    """Neo4j driver fixture"""
    if not os.getenv("RUN_DB_TESTS", "false").lower() == "true":
        pytest.skip("Database tests not enabled")
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        pytest.skip("Neo4j password not configured")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    yield driver
    
    # Cleanup
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    driver.close()


@pytest.fixture(scope="function")
def neo4j_session(neo4j_driver):
    """Neo4j session with automatic cleanup"""
    session = neo4j_driver.session()
    yield session
    session.close()


# LLM fixtures
@pytest.fixture
def ollama_client():
    """Ollama client fixture"""
    if not os.getenv("RUN_LLM_TESTS", "false").lower() == "true":
        pytest.skip("LLM tests not enabled")
    
    if not is_ollama_available():
        pytest.skip("Ollama server not available")
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Return configuration dict instead of actual client
    # Your LLM service will create the actual client
    return {
        "base_url": base_url,
        "model": os.getenv("OLLAMA_MODEL", "llama2")
    }


@pytest.fixture
def openai_config():
    """OpenAI configuration fixture"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not configured")
    
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    }


@pytest.fixture
def cohere_config():
    """Cohere configuration fixture"""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        pytest.skip("COHERE_API_KEY not configured")
    
    return {
        "api_key": api_key,
        "model": os.getenv("COHERE_MODEL", "command")
    }
