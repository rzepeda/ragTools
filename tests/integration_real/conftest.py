"""
Shared fixtures for real integration tests.

This module provides fixtures for testing with real services (PostgreSQL, Neo4j,
LM Studio, embeddings) configured via .env variables. Tests automatically skip
when services are unavailable.
"""

import os
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import requests
from sqlalchemy import create_engine, text


# =============================================================================
# Service Availability Checks
# =============================================================================


def check_postgres_available() -> bool:
    """Check if PostgreSQL is accessible."""
    url = os.getenv("DB_TEST_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not url:
        return False
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


def check_neo4j_available() -> bool:
    """Check if Neo4j is accessible."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, user, password]):
        return False
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


def check_llm_available() -> bool:
    """Check if LLM service is accessible (LM Studio or OpenAI)."""
    # Try LM Studio first
    base_url = os.getenv("LM_STUDIO_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        try:
            response = requests.get(f"{base_url.rstrip('/v1')}/v1/models", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
    
    # Check if OpenAI API key is available
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "lm-studio":
        return True
    
    return False


def check_openai_api_available() -> bool:
    """Check if OpenAI API key is available (not LM Studio)."""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key and api_key != "lm-studio" and api_key.startswith("sk-"))


def check_cohere_api_available() -> bool:
    """Check if Cohere API key is available."""
    return bool(os.getenv("COHERE_API_KEY"))


def check_embedding_model_available() -> bool:
    """Check if embedding model is configured."""
    return bool(os.getenv("EMBEDDING_MODEL_NAME"))


# =============================================================================
# Auto-Skip Fixtures
# =============================================================================


@pytest.fixture
def require_postgres():
    """Skip test if PostgreSQL not available."""
    if not check_postgres_available():
        pytest.skip(
            "PostgreSQL not available (DB_TEST_DATABASE_URL not set or unreachable)"
        )


@pytest.fixture
def require_neo4j():
    """Skip test if Neo4j not available."""
    if not check_neo4j_available():
        pytest.skip(
            "Neo4j not available (NEO4J_URI/USER/PASSWORD not set or unreachable)"
        )


@pytest.fixture
def require_llm():
    """Skip test if LLM service not available."""
    if not check_llm_available():
        pytest.skip(
            "LLM service not available (LM_STUDIO_BASE_URL or OPENAI_API_KEY not set or unreachable)"
        )


@pytest.fixture
def require_openai():
    """Skip test if OpenAI API key not available."""
    if not check_openai_api_available():
        pytest.skip("OpenAI API key not available")


@pytest.fixture
def require_cohere():
    """Skip test if Cohere API key not available."""
    if not check_cohere_api_available():
        pytest.skip("Cohere API key not available")


@pytest.fixture
def require_embeddings():
    """Skip test if embedding model not configured."""
    if not check_embedding_model_available():
        pytest.skip("Embedding model not configured (EMBEDDING_MODEL_NAME not set)")


# =============================================================================
# Real Service Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def real_db_service(require_postgres) -> AsyncGenerator:
    """Provide real PostgreSQL service."""
    from rag_factory.services.database.postgres import PostgresqlDatabaseService
    import logging
    logger = logging.getLogger(__name__)
    
    url = os.getenv("DB_TEST_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    
    # Parse URL to extract connection parameters
    # Format: postgresql://user:password@host:port/database
    from urllib.parse import urlparse
    parsed = urlparse(url)
    
    logger.info("[FIXTURE] Creating PostgresqlDatabaseService")
    service = PostgresqlDatabaseService(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password,
        table_name="test_chunks_real"
    )
    
    # Cleanup BEFORE test: ensure clean state
    # First ensure table exists by getting pool (which calls _ensure_table)
    try:
        logger.info("[FIXTURE] Getting pool for cleanup")
        pool = await service._get_pool()
        logger.info("[FIXTURE] Pool acquired, truncating table")
        # Now truncate to clean any existing data
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {service.table_name} CASCADE")
        logger.info("[FIXTURE] Table truncated successfully")
    except Exception as e:
        # Table might not exist yet or other issue, that's okay
        # The _get_pool() call will create it if needed
        logger.info(f"[FIXTURE] Cleanup failed (expected on first run): {e}")
        pass
    
    logger.info("[FIXTURE] Yielding service to test")
    yield service
    
    # Cleanup AFTER test: drop test table and close connections
    logger.info("[FIXTURE] Test completed, starting cleanup")
    try:
        pool = await service._get_pool()
        logger.info("[FIXTURE] Pool acquired for cleanup")
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {service.table_name} CASCADE")
        logger.info("[FIXTURE] Table dropped successfully")
    except Exception as e:
        logger.info(f"[FIXTURE] Cleanup failed: {e}")
        pass
    
    logger.info("[FIXTURE] Closing service")
    await service.close()
    logger.info("[FIXTURE] Service closed")



@pytest.fixture
def real_embedding_service(require_embeddings):
    """Provide real ONNX embedding service with async wrapper."""
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "models/embeddings")
    
    config = {
        "model": model_name,
        "cache_dir": model_path
    }
    
    provider = ONNXLocalProvider(config=config)
    
    # Create async wrapper
    class AsyncEmbeddingWrapper:
        def __init__(self, provider):
            self.provider = provider
        
        async def embed(self, text: str):
            """Async wrapper for single text embedding."""
            result = self.provider.get_embeddings([text])
            return result.embeddings[0]
        
        async def embed_batch(self, texts: list):
            """Async wrapper for batch embedding."""
            result = self.provider.get_embeddings(texts)
            return result.embeddings
        
        def get_dimension(self):
            """Get embedding dimension."""
            return self.provider.get_dimensions()
    
    return AsyncEmbeddingWrapper(provider)


@pytest.fixture
def real_openai_embedding_service(require_openai):
    """Provide real OpenAI embedding service."""
    from rag_factory.services.embedding.providers.openai import OpenAIEmbeddingProvider
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model="text-embedding-ada-002"
    )


@pytest.fixture
def real_cohere_embedding_service(require_cohere):
    """Provide real Cohere embedding service."""
    from rag_factory.services.embedding.providers.cohere import CohereEmbeddingProvider
    
    api_key = os.getenv("COHERE_API_KEY")
    
    return CohereEmbeddingProvider(
        api_key=api_key,
        model="embed-english-v3.0"
    )


@pytest.fixture
def real_llm_service(require_llm):
    """Provide real LLM service (LM Studio or OpenAI)."""
    from rag_factory.services.llm.service import LLMService
    from rag_factory.services.llm.config import LLMServiceConfig
    
    # Check if using real OpenAI API
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    
    if api_key != "lm-studio" and api_key.startswith("sk-"):
        # Real OpenAI API
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        config = LLMServiceConfig(
            provider="openai",
            provider_config={
                "api_key": api_key,
                "model": model
            }
        )
    else:
        # LM Studio (OpenAI-compatible)
        base_url = os.getenv("LM_STUDIO_BASE_URL") or os.getenv("OPENAI_API_BASE")
        model = os.getenv("LM_STUDIO_MODEL") or os.getenv("OPENAI_MODEL", "default")
        
        config = LLMServiceConfig(
            provider="openai",
            provider_config={
                "api_key": "lm-studio",
                "model": model,
                "base_url": base_url
            }
        )
    
    return LLMService(config)


@pytest_asyncio.fixture
async def real_neo4j_service(require_neo4j) -> AsyncGenerator:
    """Provide real Neo4j service."""
    from rag_factory.services.database.neo4j import Neo4jDatabaseService
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    service = Neo4jDatabaseService(
        uri=uri,
        user=user,
        password=password
    )
    
    yield service
    
    # Cleanup: delete test data
    try:
        await service.execute_query(
            "MATCH (n:TestNode) DETACH DELETE n"
        )
    except Exception:
        pass
    
    await service.close()


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",
        "Python is a high-level programming language",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand human language"
    ]


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    # Return simple dictionaries for testing, not database models
    return [
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "metadata": {"source": "test1.txt", "category": "animals"}
        },
        {
            "text": "Python is a high-level programming language used for web development, data science, and automation",
            "metadata": {"source": "test2.txt", "category": "programming"}
        },
        {
            "text": "Machine learning algorithms can learn from data and make predictions without being explicitly programmed",
            "metadata": {"source": "test3.txt", "category": "ai"}
        }
    ]

