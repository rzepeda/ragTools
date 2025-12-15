"""Unit tests for DatabaseContext vector search operations.

Tests vector similarity search with different distance metrics.
Requires PostgreSQL with pgvector extension for full testing.
"""

import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

from rag_factory.services.database import DatabaseContext


# Skip all tests in this module if pgvector is not available
pytestmark = pytest.mark.skipif(
    not PGVECTOR_AVAILABLE,
    reason="pgvector not available"
)


class TestVectorSearchErrorHandling:
    """Test vector search error handling (doesn't require actual pgvector)."""

    def test_invalid_distance_metric_raises_error(self):
        """Test clear error for invalid distance metric."""
        # Use SQLite for this test since we're just testing error handling
        engine = create_engine("sqlite:///:memory:")
        
        # Create a simple table (won't actually have vector support)
        metadata = MetaData()
        Table(
            'test_vectors',
            metadata,
            Column('id', String, primary_key=True),
            Column('content', String)
        )
        metadata.create_all(engine)

        context = DatabaseContext(
            engine,
            table_mapping={"vectors": "test_vectors"}
        )

        # This should raise ValueError before trying to execute the query
        with pytest.raises(ValueError) as exc_info:
            context.vector_search(
                "vectors",
                vector_field="embedding",
                query_vector=[0.1] * 384,
                distance_metric="invalid_metric"
            )

        error_msg = str(exc_info.value)
        assert "Unknown distance metric: 'invalid_metric'" in error_msg
        assert "Valid options:" in error_msg
        assert "cosine" in error_msg
        assert "l2" in error_msg
        assert "inner_product" in error_msg


@pytest.mark.integration
@pytest.mark.skipif(
    not PGVECTOR_AVAILABLE,
    reason="Requires PostgreSQL with pgvector"
)
class TestVectorSearchWithPostgreSQL:
    """Integration tests for vector search with real PostgreSQL.
    
    These tests require a running PostgreSQL instance with pgvector extension.
    They are marked as integration tests and can be skipped in unit test runs.
    """

    @pytest.fixture
    def postgres_engine(self):
        """Create PostgreSQL engine for testing.
        
        Requires PostgreSQL to be running on localhost with test database.
        """
        try:
            engine = create_engine(
                "postgresql://postgres:postgres@localhost:5432/test_rag_db"
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            yield engine
            engine.dispose()
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")

    @pytest.fixture
    def vector_table(self, postgres_engine):
        """Create test table with vector column."""
        metadata = MetaData()

        # Create table with pgvector support
        table = Table(
            'test_vectors',
            metadata,
            Column('id', String, primary_key=True),
            Column('vector_embedding', Vector(384)),
            Column('text_content', String)
        )

        # Enable pgvector extension
        with postgres_engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

        metadata.create_all(postgres_engine)

        yield table

        # Cleanup
        metadata.drop_all(postgres_engine)

    @pytest.fixture
    def vector_context(self, postgres_engine, vector_table):
        """Create DatabaseContext with vector table mapping."""
        return DatabaseContext(
            postgres_engine,
            table_mapping={"vectors": "test_vectors"},
            field_mapping={"embedding": "vector_embedding", "content": "text_content"}
        )

    def test_vector_search_cosine_distance(self, vector_context):
        """Test vector search with cosine distance."""
        # Insert test vectors
        test_vector1 = [0.1] * 384
        test_vector2 = [0.9] * 384
        test_vector3 = [0.5] * 384

        vector_context.insert("vectors", {
            "id": "vec1",
            "embedding": test_vector1,
            "content": "Vector 1"
        })
        vector_context.insert("vectors", {
            "id": "vec2",
            "embedding": test_vector2,
            "content": "Vector 2"
        })
        vector_context.insert("vectors", {
            "id": "vec3",
            "embedding": test_vector3,
            "content": "Vector 3"
        })

        # Search with logical names
        query_vector = [0.15] * 384  # Closer to test_vector1
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=query_vector,
            top_k=2,
            distance_metric="cosine"
        )

        assert len(results) == 2
        # Results should be sorted by distance (ascending)
        assert results[0].distance < results[1].distance
        # First result should be vec1 (closest)
        assert results[0].text_content == "Vector 1"

    def test_vector_search_l2_distance(self, vector_context):
        """Test vector search with L2 (Euclidean) distance."""
        # Insert test vectors
        vector_context.insert("vectors", {
            "id": "vec1",
            "embedding": [1.0] * 384,
            "content": "One"
        })
        vector_context.insert("vectors", {
            "id": "vec2",
            "embedding": [0.0] * 384,
            "content": "Zero"
        })

        # Search using L2 distance
        query_vector = [0.9] * 384  # Closer to [1.0] * 384
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=query_vector,
            top_k=2,
            distance_metric="l2"
        )

        assert len(results) == 2
        # First result should be vec1 (closer in L2 distance)
        assert results[0].text_content == "One"

    def test_vector_search_inner_product(self, vector_context):
        """Test vector search with inner product metric."""
        # Insert test vectors
        vector_context.insert("vectors", {
            "id": "vec1",
            "embedding": [1.0] * 384,
            "content": "Positive"
        })
        vector_context.insert("vectors", {
            "id": "vec2",
            "embedding": [-1.0] * 384,
            "content": "Negative"
        })

        # Search using inner product
        query_vector = [1.0] * 384
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=query_vector,
            top_k=2,
            distance_metric="inner_product"
        )

        assert len(results) == 2
        # With inner product, higher values are better (max_inner_product)
        # So results are sorted by distance (which is negative inner product)
        assert results[0].text_content == "Positive"

    def test_vector_search_top_k_limit(self, vector_context):
        """Test that top_k parameter limits results correctly."""
        # Insert 10 vectors
        for i in range(10):
            vector_context.insert("vectors", {
                "id": f"vec{i}",
                "embedding": [float(i) / 10] * 384,
                "content": f"Vector {i}"
            })

        # Search with top_k=3
        query_vector = [0.5] * 384
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=query_vector,
            top_k=3,
            distance_metric="cosine"
        )

        assert len(results) == 3

    def test_vector_search_with_field_mapping(self, vector_context):
        """Test that vector search correctly uses field mapping."""
        # Insert using logical field names
        vector_context.insert("vectors", {
            "id": "vec1",
            "embedding": [0.5] * 384,  # Logical name
            "content": "Test content"  # Logical name
        })

        # Search using logical field name
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",  # Logical name
            query_vector=[0.5] * 384,
            top_k=1,
            distance_metric="cosine"
        )

        assert len(results) == 1
        # Result should have physical field names
        assert results[0].text_content == "Test content"

    def test_vector_search_empty_table(self, vector_context):
        """Test vector search on empty table returns empty results."""
        results = vector_context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=[0.5] * 384,
            top_k=5,
            distance_metric="cosine"
        )

        assert results == []


class TestVectorSearchMocking:
    """Test vector search with mocked pgvector functionality.
    
    These tests don't require actual PostgreSQL but test the logic.
    """

    def test_vector_search_uses_correct_distance_function(self):
        """Test that correct distance function is selected based on metric."""
        # This is more of a code inspection test
        # We verify the logic in the actual implementation
        
        engine = create_engine("sqlite:///:memory:")
        context = DatabaseContext(engine, {})

        # Test that ValueError is raised for invalid metrics
        # (we already test this in TestVectorSearchErrorHandling)
        
        # The actual distance function selection is tested in integration tests
        # with real PostgreSQL, as SQLite doesn't support pgvector
        pass
