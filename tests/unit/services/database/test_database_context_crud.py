"""Unit tests for DatabaseContext CRUD operations.

Tests insert, query, update, and delete operations with field mapping.
"""

import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from rag_factory.services.database import DatabaseContext


@pytest.fixture
def test_db_with_tables():
    """Create in-memory SQLite DB with test tables."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Create physical table with mapped field names
    Table(
        'physical_chunks',
        metadata,
        Column('chunk_id', String, primary_key=True),
        Column('text_content', String),
        Column('document_id', String),
        Column('chunk_index', Integer)
    )

    metadata.create_all(engine)
    return engine


@pytest.fixture
def context_with_mapping(test_db_with_tables):
    """Create DatabaseContext with field mappings."""
    return DatabaseContext(
        test_db_with_tables,
        table_mapping={"chunks": "physical_chunks"},
        field_mapping={
            "content": "text_content",
            "doc_id": "document_id"
        }
    )


class TestInsertOperation:
    """Test insert operations with field mapping."""

    def test_insert_with_field_mapping(self, context_with_mapping):
        """Test insert maps logical fields to physical fields."""
        # Insert with logical names
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Hello world",  # Logical name
            "doc_id": "doc123",       # Logical name
            "chunk_index": 0
        })

        # Verify data inserted with physical field names
        results = context_with_mapping.query("chunks", filters={"doc_id": "doc123"})
        assert len(results) == 1
        assert results[0].text_content == "Hello world"  # Physical name
        assert results[0].document_id == "doc123"        # Physical name

    def test_insert_without_field_mapping(self, test_db_with_tables):
        """Test insert works with unmapped fields."""
        context = DatabaseContext(
            test_db_with_tables,
            table_mapping={"chunks": "physical_chunks"}
        )

        # Insert with physical names directly
        context.insert("chunks", {
            "chunk_id": "chunk1",
            "text_content": "Test",
            "document_id": "doc1",
            "chunk_index": 0
        })

        results = context.query("chunks")
        assert len(results) == 1
        assert results[0].text_content == "Test"

    def test_insert_multiple_rows(self, context_with_mapping):
        """Test inserting multiple rows."""
        for i in range(3):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": f"Content {i}",
                "doc_id": "doc1",
                "chunk_index": i
            })

        results = context_with_mapping.query("chunks")
        assert len(results) == 3


class TestQueryOperation:
    """Test query operations with filters."""

    def test_query_all_rows(self, context_with_mapping):
        """Test querying all rows without filters."""
        # Insert test data
        for i in range(3):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": f"Test {i}",
                "doc_id": "doc1",
                "chunk_index": i
            })

        results = context_with_mapping.query("chunks")
        assert len(results) == 3

    def test_query_with_filter(self, context_with_mapping):
        """Test query applies filter mapping correctly."""
        # Insert test data
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Test 1",
            "doc_id": "doc123",
            "chunk_index": 0
        })
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk2",
            "content": "Test 2",
            "doc_id": "doc456",
            "chunk_index": 1
        })

        # Query with logical field name
        results = context_with_mapping.query("chunks", filters={"doc_id": "doc123"})

        assert len(results) == 1
        assert results[0].chunk_id == "chunk1"

    def test_query_with_multiple_filters(self, context_with_mapping):
        """Test query with multiple filter conditions."""
        # Insert test data
        for i in range(5):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": f"Test {i}",
                "doc_id": "doc1" if i < 3 else "doc2",
                "chunk_index": i
            })

        # Query with multiple filters
        results = context_with_mapping.query(
            "chunks",
            filters={"doc_id": "doc1", "chunk_index": 1}
        )

        assert len(results) == 1
        assert results[0].chunk_id == "chunk1"

    def test_query_with_limit(self, context_with_mapping):
        """Test query limit parameter."""
        # Insert test data
        for i in range(10):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": f"Test {i}",
                "doc_id": "doc1",
                "chunk_index": i
            })

        results = context_with_mapping.query("chunks", limit=5)
        assert len(results) == 5

    def test_query_empty_result(self, context_with_mapping):
        """Test query returns empty list when no matches."""
        results = context_with_mapping.query(
            "chunks",
            filters={"doc_id": "nonexistent"}
        )
        assert results == []


class TestUpdateOperation:
    """Test update operations with field mapping."""

    def test_update_with_mapping(self, context_with_mapping):
        """Test update maps both filters and updates."""
        # Insert
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Original",
            "doc_id": "doc123",
            "chunk_index": 0
        })

        # Update with logical names
        context_with_mapping.update(
            "chunks",
            filters={"doc_id": "doc123"},
            updates={"content": "Updated"}
        )

        # Verify
        results = context_with_mapping.query("chunks", filters={"doc_id": "doc123"})
        assert results[0].text_content == "Updated"

    def test_update_multiple_rows(self, context_with_mapping):
        """Test updating multiple rows at once."""
        # Insert multiple rows with same doc_id
        for i in range(3):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": "Original",
                "doc_id": "doc1",
                "chunk_index": i
            })

        # Update all rows with doc_id = doc1
        context_with_mapping.update(
            "chunks",
            filters={"doc_id": "doc1"},
            updates={"content": "Updated"}
        )

        results = context_with_mapping.query("chunks", filters={"doc_id": "doc1"})
        assert len(results) == 3
        for row in results:
            assert row.text_content == "Updated"

    def test_update_multiple_fields(self, context_with_mapping):
        """Test updating multiple fields at once."""
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Original",
            "doc_id": "doc1",
            "chunk_index": 0
        })

        context_with_mapping.update(
            "chunks",
            filters={"chunk_id": "chunk1"},
            updates={"content": "New content", "chunk_index": 99}
        )

        results = context_with_mapping.query("chunks", filters={"chunk_id": "chunk1"})
        assert results[0].text_content == "New content"
        assert results[0].chunk_index == 99


class TestDeleteOperation:
    """Test delete operations with field mapping."""

    def test_delete_with_mapping(self, context_with_mapping):
        """Test delete maps filter fields correctly."""
        # Insert
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Test",
            "doc_id": "doc123",
            "chunk_index": 0
        })

        # Delete with logical name
        context_with_mapping.delete("chunks", filters={"doc_id": "doc123"})

        # Verify deleted
        results = context_with_mapping.query("chunks")
        assert len(results) == 0

    def test_delete_multiple_rows(self, context_with_mapping):
        """Test deleting multiple rows with same filter."""
        # Insert multiple rows
        for i in range(3):
            context_with_mapping.insert("chunks", {
                "chunk_id": f"chunk{i}",
                "content": f"Test {i}",
                "doc_id": "doc1",
                "chunk_index": i
            })

        # Delete all rows with doc_id = doc1
        context_with_mapping.delete("chunks", filters={"doc_id": "doc1"})

        results = context_with_mapping.query("chunks")
        assert len(results) == 0

    def test_delete_with_specific_filter(self, context_with_mapping):
        """Test delete with specific filter only deletes matching rows."""
        # Insert multiple rows
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Test 1",
            "doc_id": "doc1",
            "chunk_index": 0
        })
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk2",
            "content": "Test 2",
            "doc_id": "doc2",
            "chunk_index": 1
        })

        # Delete only doc1
        context_with_mapping.delete("chunks", filters={"doc_id": "doc1"})

        # Verify only doc2 remains
        results = context_with_mapping.query("chunks")
        assert len(results) == 1
        assert results[0].document_id == "doc2"


class TestCRUDIntegration:
    """Test CRUD operations working together."""

    def test_full_crud_cycle(self, context_with_mapping):
        """Test complete CRUD cycle: Create, Read, Update, Delete."""
        # Create
        context_with_mapping.insert("chunks", {
            "chunk_id": "chunk1",
            "content": "Original content",
            "doc_id": "doc1",
            "chunk_index": 0
        })

        # Read
        results = context_with_mapping.query("chunks", filters={"chunk_id": "chunk1"})
        assert len(results) == 1
        assert results[0].text_content == "Original content"

        # Update
        context_with_mapping.update(
            "chunks",
            filters={"chunk_id": "chunk1"},
            updates={"content": "Updated content"}
        )

        # Read again
        results = context_with_mapping.query("chunks", filters={"chunk_id": "chunk1"})
        assert results[0].text_content == "Updated content"

        # Delete
        context_with_mapping.delete("chunks", filters={"chunk_id": "chunk1"})

        # Verify deleted
        results = context_with_mapping.query("chunks")
        assert len(results) == 0
