"""Unit tests for DatabaseContext class.

Tests the core functionality of DatabaseContext including initialization,
table/field mapping, and error handling.
"""

import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from rag_factory.services.database import DatabaseContext


class TestDatabaseContextInitialization:
    """Test DatabaseContext initialization."""

    def test_initialization_with_all_parameters(self):
        """Test DatabaseContext initializes with correct parameters."""
        engine = create_engine("sqlite:///:memory:")
        table_mapping = {"chunks": "physical_chunks", "vectors": "physical_vectors"}
        field_mapping = {"content": "text_content", "embedding": "vector_embedding"}

        context = DatabaseContext(engine, table_mapping, field_mapping)

        assert context.engine is engine
        assert context.tables == table_mapping
        assert context.fields == field_mapping
        assert context._metadata is not None
        assert context._reflected_tables == {}

    def test_initialization_without_field_mapping(self):
        """Test DatabaseContext works without field mapping."""
        engine = create_engine("sqlite:///:memory:")
        table_mapping = {"chunks": "physical_chunks"}

        context = DatabaseContext(engine, table_mapping)

        assert context.fields == {}
        assert context._map_field("content") == "content"  # Pass-through

    def test_field_mapping_passthrough(self):
        """Test unmapped fields pass through unchanged."""
        engine = create_engine("sqlite:///:memory:")
        field_mapping = {"content": "text_content"}

        context = DatabaseContext(engine, {}, field_mapping)

        assert context._map_field("content") == "text_content"
        assert context._map_field("unmapped_field") == "unmapped_field"


class TestTableMapping:
    """Test table name mapping and reflection."""

    def test_get_table_unmapped_logical_name_raises_error(self):
        """Test clear error when logical table name not in mapping."""
        engine = create_engine("sqlite:///:memory:")
        table_mapping = {"chunks": "physical_chunks", "vectors": "physical_vectors"}

        context = DatabaseContext(engine, table_mapping)

        with pytest.raises(KeyError) as exc_info:
            context.get_table("nonexistent")

        error_msg = str(exc_info.value)
        assert "No table mapping for 'nonexistent'" in error_msg
        assert "chunks" in error_msg
        assert "vectors" in error_msg

    def test_get_table_caches_reflected_tables(self):
        """Test that reflected tables are cached."""
        engine = create_engine("sqlite:///:memory:")
        
        # Create a physical table
        metadata = MetaData()
        Table(
            'physical_chunks',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('text', String)
        )
        metadata.create_all(engine)

        context = DatabaseContext(engine, {"chunks": "physical_chunks"})

        # First call should reflect and cache
        table1 = context.get_table("chunks")
        assert "physical_chunks" in context._reflected_tables

        # Second call should return cached table
        table2 = context.get_table("chunks")
        assert table1 is table2  # Same object


class TestFieldMapping:
    """Test field name mapping."""

    def test_map_field_with_mapping(self):
        """Test field mapping with defined mappings."""
        engine = create_engine("sqlite:///:memory:")
        field_mapping = {
            "content": "text_content",
            "doc_id": "document_id",
            "embedding": "vector_embedding"
        }

        context = DatabaseContext(engine, {}, field_mapping)

        assert context._map_field("content") == "text_content"
        assert context._map_field("doc_id") == "document_id"
        assert context._map_field("embedding") == "vector_embedding"

    def test_map_field_without_mapping(self):
        """Test field mapping returns original name when no mapping exists."""
        engine = create_engine("sqlite:///:memory:")
        field_mapping = {"content": "text_content"}

        context = DatabaseContext(engine, {}, field_mapping)

        # Mapped field
        assert context._map_field("content") == "text_content"
        
        # Unmapped fields should pass through
        assert context._map_field("doc_id") == "doc_id"
        assert context._map_field("chunk_index") == "chunk_index"
        assert context._map_field("metadata") == "metadata"

    def test_map_field_with_empty_mapping(self):
        """Test field mapping with no mappings defined."""
        engine = create_engine("sqlite:///:memory:")

        context = DatabaseContext(engine, {})

        # All fields should pass through
        assert context._map_field("content") == "content"
        assert context._map_field("doc_id") == "doc_id"
        assert context._map_field("any_field") == "any_field"


class TestContextIsolation:
    """Test that multiple contexts are properly isolated."""

    def test_different_contexts_have_different_mappings(self):
        """Test that different contexts maintain separate mappings."""
        engine = create_engine("sqlite:///:memory:")

        context1 = DatabaseContext(
            engine,
            table_mapping={"chunks": "semantic_chunks"},
            field_mapping={"content": "text_content"}
        )

        context2 = DatabaseContext(
            engine,
            table_mapping={"chunks": "keyword_chunks"},
            field_mapping={"content": "raw_text"}
        )

        # Verify they have different mappings
        assert context1.tables["chunks"] == "semantic_chunks"
        assert context2.tables["chunks"] == "keyword_chunks"
        assert context1._map_field("content") == "text_content"
        assert context2._map_field("content") == "raw_text"

    def test_contexts_share_same_engine(self):
        """Test that multiple contexts can share the same engine."""
        engine = create_engine("sqlite:///:memory:")

        context1 = DatabaseContext(engine, {"chunks": "table1"})
        context2 = DatabaseContext(engine, {"chunks": "table2"})

        # Both should reference the same engine
        assert context1.engine is context2.engine
        assert context1.engine is engine

    def test_contexts_have_independent_reflection_cache(self):
        """Test that contexts maintain independent table reflection caches."""
        engine = create_engine("sqlite:///:memory:")
        
        # Create two physical tables
        metadata = MetaData()
        Table('table1', metadata, Column('id', Integer, primary_key=True))
        Table('table2', metadata, Column('id', Integer, primary_key=True))
        metadata.create_all(engine)

        context1 = DatabaseContext(engine, {"chunks": "table1"})
        context2 = DatabaseContext(engine, {"chunks": "table2"})

        # Reflect tables
        context1.get_table("chunks")
        context2.get_table("chunks")

        # Each context should have its own cache
        assert "table1" in context1._reflected_tables
        assert "table2" in context2._reflected_tables
        assert "table2" not in context1._reflected_tables
        assert "table1" not in context2._reflected_tables
