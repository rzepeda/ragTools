"""Integration tests for multi-context isolation.

Tests that multiple DatabaseContext instances can coexist on the same
database without interfering with each other while sharing connection pool.

These tests require a running PostgreSQL instance.
"""

import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, text

from rag_factory.services.database import PostgresqlDatabaseService, DatabaseContext


@pytest.fixture
def postgres_connection_string():
    """PostgreSQL connection string for testing."""
    return "postgresql://postgres:postgres@localhost:5432/test_rag_db"


@pytest.fixture
def postgres_engine(postgres_connection_string):
    """Create PostgreSQL engine for testing."""
    try:
        engine = create_engine(postgres_connection_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        yield engine
        engine.dispose()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.fixture
def test_tables(postgres_engine):
    """Create test tables for multi-context testing."""
    metadata = MetaData()

    # Create tables for semantic search strategy
    semantic_chunks = Table(
        'semantic_chunks',
        metadata,
        Column('chunk_id', String, primary_key=True),
        Column('text_content', String),
        Column('document_id', String),
        Column('chunk_index', Integer)
    )

    # Create tables for keyword search strategy
    keyword_chunks = Table(
        'keyword_chunks',
        metadata,
        Column('chunk_id', String, primary_key=True),
        Column('content', String),
        Column('doc_id', String),
        Column('index', Integer)
    )

    # Create all tables
    metadata.create_all(postgres_engine)

    yield {
        'semantic_chunks': semantic_chunks,
        'keyword_chunks': keyword_chunks
    }

    # Cleanup
    metadata.drop_all(postgres_engine)


@pytest.mark.integration
class TestMultiContextIsolation:
    """Test multiple contexts operating on same database."""

    def test_contexts_share_same_engine(self, postgres_engine):
        """Test multiple contexts share the same engine."""
        context1 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"}
        )
        context2 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"}
        )

        # Verify they share the same engine
        assert context1.engine is context2.engine
        assert context1.engine is postgres_engine

    def test_contexts_isolated_different_tables(self, postgres_engine, test_tables):
        """Test contexts write to different physical tables."""
        # Context for semantic search
        semantic_ctx = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"},
            field_mapping={"content": "text_content", "doc_id": "document_id"}
        )

        # Context for keyword search
        keyword_ctx = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"},
            field_mapping={"content": "content", "doc_id": "doc_id"}
        )

        # Both insert to logical "chunks" table
        semantic_ctx.insert("chunks", {
            "chunk_id": "s1",
            "content": "Semantic content",
            "doc_id": "doc1",
            "chunk_index": 0
        })

        keyword_ctx.insert("chunks", {
            "chunk_id": "k1",
            "content": "Keyword content",
            "doc_id": "doc1",
            "index": 0
        })

        # Verify isolation - each context only sees its own data
        semantic_results = semantic_ctx.query("chunks")
        keyword_results = keyword_ctx.query("chunks")

        assert len(semantic_results) == 1
        assert len(keyword_results) == 1
        assert semantic_results[0].chunk_id == "s1"
        assert keyword_results[0].chunk_id == "k1"

    def test_contexts_with_same_logical_names_different_physical(
        self, postgres_engine, test_tables
    ):
        """Test contexts can use same logical names for different physical tables."""
        ctx1 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"}
        )
        ctx2 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"}
        )

        # Insert to "chunks" in both contexts
        ctx1.insert("chunks", {
            "chunk_id": "c1",
            "text_content": "Text 1",
            "document_id": "doc1",
            "chunk_index": 0
        })

        ctx2.insert("chunks", {
            "chunk_id": "c2",
            "content": "Text 2",
            "doc_id": "doc2",
            "index": 0
        })

        # Each context only sees its own data
        assert len(ctx1.query("chunks")) == 1
        assert len(ctx2.query("chunks")) == 1

        # Verify they're in different physical tables
        assert ctx1.query("chunks")[0].chunk_id == "c1"
        assert ctx2.query("chunks")[0].chunk_id == "c2"

    def test_concurrent_operations_on_different_contexts(
        self, postgres_engine, test_tables
    ):
        """Test concurrent operations on different contexts don't interfere."""
        ctx1 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"}
        )
        ctx2 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"}
        )

        # Interleave operations
        ctx1.insert("chunks", {
            "chunk_id": "s1",
            "text_content": "Semantic 1",
            "document_id": "doc1",
            "chunk_index": 0
        })

        ctx2.insert("chunks", {
            "chunk_id": "k1",
            "content": "Keyword 1",
            "doc_id": "doc1",
            "index": 0
        })

        ctx1.insert("chunks", {
            "chunk_id": "s2",
            "text_content": "Semantic 2",
            "document_id": "doc1",
            "chunk_index": 1
        })

        ctx2.insert("chunks", {
            "chunk_id": "k2",
            "content": "Keyword 2",
            "doc_id": "doc1",
            "index": 1
        })

        # Verify each context has correct data
        assert len(ctx1.query("chunks")) == 2
        assert len(ctx2.query("chunks")) == 2

    def test_update_in_one_context_doesnt_affect_other(
        self, postgres_engine, test_tables
    ):
        """Test updates in one context don't affect other contexts."""
        ctx1 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"}
        )
        ctx2 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"}
        )

        # Insert data in both
        ctx1.insert("chunks", {
            "chunk_id": "c1",
            "text_content": "Original 1",
            "document_id": "doc1",
            "chunk_index": 0
        })

        ctx2.insert("chunks", {
            "chunk_id": "c1",  # Same ID but different table
            "content": "Original 2",
            "doc_id": "doc1",
            "index": 0
        })

        # Update in ctx1
        ctx1.update(
            "chunks",
            filters={"chunk_id": "c1"},
            updates={"text_content": "Updated 1"}
        )

        # Verify ctx1 updated but ctx2 unchanged
        assert ctx1.query("chunks")[0].text_content == "Updated 1"
        assert ctx2.query("chunks")[0].content == "Original 2"

    def test_delete_in_one_context_doesnt_affect_other(
        self, postgres_engine, test_tables
    ):
        """Test deletes in one context don't affect other contexts."""
        ctx1 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "semantic_chunks"}
        )
        ctx2 = DatabaseContext(
            postgres_engine,
            table_mapping={"chunks": "keyword_chunks"}
        )

        # Insert data in both
        ctx1.insert("chunks", {
            "chunk_id": "c1",
            "text_content": "Text 1",
            "document_id": "doc1",
            "chunk_index": 0
        })

        ctx2.insert("chunks", {
            "chunk_id": "c1",
            "content": "Text 2",
            "doc_id": "doc1",
            "index": 0
        })

        # Delete from ctx1
        ctx1.delete("chunks", filters={"chunk_id": "c1"})

        # Verify ctx1 empty but ctx2 still has data
        assert len(ctx1.query("chunks")) == 0
        assert len(ctx2.query("chunks")) == 1


@pytest.mark.integration
class TestPostgresServiceMultiContext:
    """Test PostgresqlDatabaseService with multiple contexts."""

    @pytest.fixture
    def db_service(self, postgres_connection_string):
        """Create PostgresqlDatabaseService for testing."""
        # Extract connection params from string
        service = PostgresqlDatabaseService(
            host="localhost",
            port=5432,
            database="test_rag_db",
            user="postgres",
            password="postgres"
        )
        yield service

    def test_service_creates_contexts_with_shared_engine(
        self, db_service, test_tables
    ):
        """Test service creates contexts that share the same engine."""
        ctx1 = db_service.get_context({"chunks": "semantic_chunks"})
        ctx2 = db_service.get_context({"chunks": "keyword_chunks"})

        # Both should share the same engine
        assert ctx1.engine is ctx2.engine

    def test_service_caches_contexts(self, db_service):
        """Test service caches contexts for same mappings."""
        table_mapping = {"chunks": "test_chunks"}
        field_mapping = {"content": "text"}

        ctx1 = db_service.get_context(table_mapping, field_mapping)
        ctx2 = db_service.get_context(table_mapping, field_mapping)

        # Should return same instance
        assert ctx1 is ctx2

    def test_service_creates_different_contexts_for_different_mappings(
        self, db_service
    ):
        """Test service creates different contexts for different mappings."""
        ctx1 = db_service.get_context({"chunks": "semantic_chunks"})
        ctx2 = db_service.get_context({"chunks": "keyword_chunks"})

        # Should be different instances
        assert ctx1 is not ctx2
        # But share same engine
        assert ctx1.engine is ctx2.engine

    def test_multiple_strategies_scenario(self, db_service, test_tables):
        """Test realistic scenario with multiple RAG strategies."""
        # Strategy 1: Semantic search
        semantic_ctx = db_service.get_context(
            table_mapping={"chunks": "semantic_chunks"},
            field_mapping={"content": "text_content", "doc_id": "document_id"}
        )

        # Strategy 2: Keyword search
        keyword_ctx = db_service.get_context(
            table_mapping={"chunks": "keyword_chunks"},
            field_mapping={"content": "content", "doc_id": "doc_id"}
        )

        # Both strategies index the same document
        doc_text = "Python is a programming language"

        # Semantic strategy stores with embeddings
        semantic_ctx.insert("chunks", {
            "chunk_id": "sem_1",
            "content": doc_text,
            "doc_id": "doc123",
            "chunk_index": 0
        })

        # Keyword strategy stores with inverted index
        keyword_ctx.insert("chunks", {
            "chunk_id": "key_1",
            "content": doc_text,
            "doc_id": "doc123",
            "index": 0
        })

        # Each strategy can query its own data
        sem_results = semantic_ctx.query("chunks", filters={"doc_id": "doc123"})
        key_results = keyword_ctx.query("chunks", filters={"doc_id": "doc123"})

        assert len(sem_results) == 1
        assert len(key_results) == 1
        assert sem_results[0].chunk_id == "sem_1"
        assert key_results[0].chunk_id == "key_1"

        # Verify they're using different physical tables
        assert sem_results[0].text_content == doc_text  # Physical field name
        assert key_results[0].content == doc_text  # Different physical field name
