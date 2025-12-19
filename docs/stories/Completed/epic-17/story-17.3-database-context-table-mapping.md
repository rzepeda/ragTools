# Story 17.3: Implement DatabaseContext for Table Mapping

**Story ID:** 17.3
**Epic:** Epic 17 - Strategy Pair Configuration System
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 11 (Dependency Injection), Epic 16 (Database Consolidation)

---

## User Story

**As a** developer
**I want** each strategy to access strategy-specific tables on a shared database
**So that** strategies are isolated but share connection pools efficiently

---

## Background

In Epic 16, we consolidated all database access into a single `PostgresqlDatabaseService` with connection pooling. However, multiple RAG strategies need to coexist on the same database without interfering with each other:

- **Semantic search** needs `semantic_chunks` and `semantic_vectors` tables
- **Keyword search** needs `keyword_chunks` and `keyword_inverted_index` tables
- **Graph search** needs `entities` and `relationships` tables

**The Problem:**
- All strategies use the same `PostgresqlDatabaseService` instance (same connection pool)
- But they can't all use tables with the same names (e.g., all using `chunks` table)
- Hard-coding physical table names in strategies breaks portability

**The Solution: DatabaseContext**

A `DatabaseContext` provides a strategy-specific view of the shared database:
- **Logical names** in strategy code: `context.insert("chunks", {...})`
- **Physical names** in database: Inserts into `semantic_chunks` or `keyword_chunks`
- **Same connection pool** shared across all contexts
- **Complete isolation** between strategies

---

## Detailed Requirements

### Functional Requirements

1. **DatabaseContext Class**
   - Wraps a shared SQLAlchemy engine
   - Provides table name mapping (logical → physical)
   - Provides field name mapping (logical → physical)
   - Reflects table schemas automatically
   - Caches reflected table metadata
   - Thread-safe for concurrent access

2. **Logical-to-Physical Table Translation**
   - Accept configuration like `{"chunks": "semantic_chunks"}`
   - Map logical table name to physical table in all operations
   - Raise clear error if logical name not in mapping
   - List available logical names in error messages
   - Support arbitrary mappings (not hardcoded)

3. **Logical-to-Physical Field Translation**
   - Accept configuration like `{"content": "text_content"}`
   - Map logical field names to physical fields in queries
   - Default to same name if no mapping provided
   - Apply recursively in filter conditions and updates
   - Preserve unmapped field names (pass through)

4. **CRUD Operations with Logical Names**
   - **Insert:** `context.insert("chunks", {"content": "hello"})`
     - Maps to: `INSERT INTO semantic_chunks (text_content) VALUES ('hello')`
   - **Query:** `context.query("chunks", filters={"doc_id": "123"})`
     - Maps to: `SELECT * FROM semantic_chunks WHERE document_id = '123'`
   - **Update:** `context.update("chunks", filters={...}, updates={...})`
     - Maps both filters and updates to physical fields
   - **Delete:** `context.delete("chunks", filters={...})`
     - Maps filters to physical fields

5. **Vector Search Operations**
   - Support pgvector distance functions (cosine, L2, inner product)
   - Accept logical table and field names
   - Map to physical names before executing query
   - Return results with distance scores
   - Support top_k limiting
   - Use proper indexing for performance

6. **PostgresqlDatabaseService Extension**
   - Add `get_context(table_mapping, field_mapping)` method
   - Return `DatabaseContext` instance with shared engine
   - Cache contexts to avoid recreating for same mappings
   - Maintain connection pool across all contexts
   - Handle context lifecycle properly

7. **Table Metadata Reflection**
   - Reflect table structure from database using SQLAlchemy
   - Cache reflected tables within context
   - Lazy loading (reflect on first access)
   - Handle missing tables with clear errors
   - Support schema changes during runtime

### Non-Functional Requirements

1. **Performance**
   - Context creation: <10ms (cached after first use)
   - Table reflection: <50ms per table (cached)
   - No performance penalty for logical→physical mapping (<1ms overhead)
   - Query execution same speed as direct SQLAlchemy
   - Support thousands of concurrent contexts

2. **Isolation**
   - Strategies using different contexts cannot interfere
   - Table name conflicts prevented by mapping
   - Each context has independent table cache
   - Shared engine for connection efficiency
   - Thread-safe for concurrent operations

3. **Resource Efficiency**
   - Single connection pool shared by all contexts
   - Metadata reflection cached to avoid repeated queries
   - Context instances cached by mapping signature
   - No duplication of table schemas in memory
   - Minimal memory overhead per context (<1MB)

4. **Developer Experience**
   - Clear error messages for missing mappings
   - Intuitive API matching SQLAlchemy patterns
   - Type hints for IDE support
   - Comprehensive docstrings
   - Easy debugging with logical names preserved in logs

5. **Maintainability**
   - Clean separation from PostgresqlDatabaseService
   - No breaking changes to Epic 16 interfaces
   - Backward compatible with existing database code
   - Testable in isolation with mock engines
   - Clear class responsibilities

---

## Acceptance Criteria

### AC1: DatabaseContext Class Implementation
- [ ] `DatabaseContext` class created with engine, table_mapping, field_mapping parameters
- [ ] Stores mappings as instance variables
- [ ] Initializes SQLAlchemy MetaData instance
- [ ] Maintains cache for reflected tables
- [ ] Thread-safe for concurrent access
- [ ] Clean, documented code with type hints

### AC2: Table Name Mapping
- [ ] `get_table(logical_name)` method implemented
- [ ] Maps logical names to physical table names
- [ ] Reflects table schema from database
- [ ] Caches reflected tables
- [ ] Raises `KeyError` with helpful message for unmapped names
- [ ] Lists available logical names in error

### AC3: Field Name Mapping
- [ ] `_map_field(logical_field)` method implemented
- [ ] Maps logical field names to physical names
- [ ] Returns logical name if no mapping exists (pass-through)
- [ ] Works with nested queries and filters
- [ ] Handles None values gracefully

### AC4: CRUD Operations
- [ ] `insert(logical_table, data)` working
- [ ] `query(logical_table, filters, limit)` working
- [ ] `update(logical_table, filters, updates)` working
- [ ] `delete(logical_table, filters)` working
- [ ] All operations use logical names in API
- [ ] All operations translate to physical names in SQL

### AC5: Vector Search Operations
- [ ] `vector_search(logical_table, vector_field, query_vector, top_k, distance_metric)` implemented
- [ ] Supports cosine distance metric
- [ ] Supports L2 distance metric
- [ ] Supports inner product metric
- [ ] Uses pgvector extension properly
- [ ] Returns results sorted by distance

### AC6: PostgresqlDatabaseService Extension
- [ ] `get_context(table_mapping, field_mapping)` method added
- [ ] Returns `DatabaseContext` with shared engine
- [ ] Caches contexts for same mappings
- [ ] Multiple contexts share same engine
- [ ] No breaking changes to Epic 16 API

### AC7: Multiple Contexts Isolation
- [ ] Two contexts on same DB with different table mappings work independently
- [ ] Context A inserts to `semantic_chunks`, Context B to `keyword_chunks`
- [ ] No interference between contexts
- [ ] Both use same connection pool (verified with engine identity)
- [ ] Concurrent operations safe

### AC8: Error Handling
- [ ] Clear error when logical table name not in mapping
- [ ] Clear error when physical table doesn't exist
- [ ] Clear error when invalid distance metric specified
- [ ] Helpful messages with available options
- [ ] Errors include context for debugging

### AC9: Testing
- [ ] Unit tests for `DatabaseContext` class (>90% coverage)
- [ ] Unit tests for all CRUD operations
- [ ] Unit tests for vector search operations
- [ ] Unit tests for field mapping edge cases
- [ ] Integration tests with real PostgreSQL database
- [ ] Integration tests showing multi-context isolation
- [ ] Performance benchmarks meet requirements

### AC10: Documentation
- [ ] Comprehensive docstrings for all public methods
- [ ] Usage examples in docstrings
- [ ] Integration guide with ServiceRegistry
- [ ] Example configurations showing table mappings
- [ ] Performance characteristics documented

---

## Technical Specifications

### File Structure

```
rag_factory/
├── services/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres_service.py          # PostgresqlDatabaseService (Epic 16)
│   │   ├── database_context.py          # NEW: DatabaseContext class
│   │   └── interfaces.py                # IDatabaseService interface
│   └── ...

tests/
├── unit/
│   ├── services/
│   │   ├── database/
│   │   │   ├── test_database_context.py          # NEW: Context unit tests
│   │   │   ├── test_database_context_crud.py     # NEW: CRUD operation tests
│   │   │   ├── test_database_context_vector.py   # NEW: Vector search tests
│   │   │   └── test_postgres_service_context.py  # NEW: Integration tests
│   └── ...
│
└── integration/
    └── database/
        └── test_multi_context_isolation.py       # NEW: Multi-context tests
```

### Dependencies

```python
# requirements.txt - No new dependencies needed
# Uses existing:
# - sqlalchemy>=2.0.0 (from Epic 16)
# - psycopg2-binary>=2.9.0 (from Epic 16)
# - pgvector>=0.2.0 (from Epic 16)
```

### Implementation

#### DatabaseContext Class

```python
# rag_factory/services/database/database_context.py
"""
Database context providing strategy-specific view of shared database.
"""
from typing import Dict, Optional, List, Any, Tuple
from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine


class DatabaseContext:
    """
    Strategy-specific view of a shared database.

    Provides access to tables using logical names that map to
    physical table names, enabling strategy isolation on shared database.

    Example:
        >>> db_service = registry.get("db1")
        >>> context = db_service.get_context(
        ...     table_mapping={"chunks": "semantic_chunks"},
        ...     field_mapping={"content": "text_content"}
        ... )
        >>> context.insert("chunks", {"content": "hello", "doc_id": "123"})
        # Inserts into semantic_chunks.text_content
    """

    def __init__(
        self,
        engine: Engine,
        table_mapping: Dict[str, str],
        field_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Create database context with table/field mappings.

        Args:
            engine: SQLAlchemy engine (shared across contexts)
            table_mapping: Dict mapping logical → physical table names
                          e.g., {"chunks": "semantic_chunks", "vectors": "semantic_vectors"}
            field_mapping: Optional dict mapping logical → physical field names
                          e.g., {"content": "text_content", "embedding": "vector_embedding"}
        """
        self.engine = engine
        self.tables = table_mapping
        self.fields = field_mapping or {}
        self._metadata = MetaData()
        self._reflected_tables: Dict[str, Table] = {}

    def get_table(self, logical_name: str) -> Table:
        """
        Get SQLAlchemy Table object by logical name.

        Args:
            logical_name: Logical table name like "chunks" or "vectors"

        Returns:
            Reflected Table object for physical table

        Raises:
            KeyError: If logical_name not in table_mapping

        Example:
            >>> table = context.get_table("chunks")
            >>> # Returns Table object for "semantic_chunks" physical table
        """
        if logical_name not in self.tables:
            available = list(self.tables.keys())
            raise KeyError(
                f"No table mapping for '{logical_name}'. "
                f"Available logical names: {available}"
            )

        physical_name = self.tables[logical_name]

        # Reflect table structure if not cached
        if physical_name not in self._reflected_tables:
            self._reflected_tables[physical_name] = Table(
                physical_name,
                self._metadata,
                autoload_with=self.engine
            )

        return self._reflected_tables[physical_name]

    def _map_field(self, logical_field: str) -> str:
        """
        Map logical field name to physical field name.

        Args:
            logical_field: Logical field name like "content" or "doc_id"

        Returns:
            Physical field name, or logical name if no mapping exists

        Example:
            >>> context._map_field("content")  # With mapping {"content": "text_content"}
            'text_content'
            >>> context._map_field("doc_id")   # No mapping
            'doc_id'
        """
        return self.fields.get(logical_field, logical_field)

    def insert(self, logical_table: str, data: Dict[str, Any]) -> None:
        """
        Insert row into a logically-named table.

        Args:
            logical_table: Logical table name (e.g., "chunks")
            data: Dict with logical field names as keys

        Example:
            >>> context.insert("chunks", {
            ...     "content": "Hello world",
            ...     "doc_id": "doc123",
            ...     "chunk_index": 0
            ... })
            # Inserts into physical table with mapped field names
        """
        table = self.get_table(logical_table)

        # Map logical field names to physical field names
        physical_data = {
            self._map_field(k): v for k, v in data.items()
        }

        with self.engine.begin() as conn:
            conn.execute(insert(table).values(**physical_data))

    def query(
        self,
        logical_table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Query a logically-named table.

        Args:
            logical_table: Logical table name
            filters: Optional dict of logical_field → value for WHERE clause
            limit: Optional row limit

        Returns:
            List of row results

        Example:
            >>> results = context.query(
            ...     "chunks",
            ...     filters={"doc_id": "doc123"},
            ...     limit=10
            ... )
            # Queries physical table with mapped field names
        """
        table = self.get_table(logical_table)
        query = select(table)

        # Apply filters (map logical to physical field names)
        if filters:
            for logical_field, value in filters.items():
                physical_field = self._map_field(logical_field)
                query = query.where(table.c[physical_field] == value)

        if limit:
            query = query.limit(limit)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchall()

    def update(
        self,
        logical_table: str,
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> None:
        """
        Update rows in a logically-named table.

        Args:
            logical_table: Logical table name
            filters: Dict of logical_field → value for WHERE clause
            updates: Dict of logical_field → new_value for SET clause

        Example:
            >>> context.update(
            ...     "chunks",
            ...     filters={"doc_id": "doc123"},
            ...     updates={"content": "Updated text"}
            ... )
            # Updates physical table with mapped field names
        """
        table = self.get_table(logical_table)
        stmt = update(table)

        # Apply WHERE filters
        for logical_field, value in filters.items():
            physical_field = self._map_field(logical_field)
            stmt = stmt.where(table.c[physical_field] == value)

        # Apply SET updates
        physical_updates = {
            self._map_field(k): v for k, v in updates.items()
        }
        stmt = stmt.values(**physical_updates)

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def delete(self, logical_table: str, filters: Dict[str, Any]) -> None:
        """
        Delete rows from a logically-named table.

        Args:
            logical_table: Logical table name
            filters: Dict of logical_field → value for WHERE clause

        Example:
            >>> context.delete("chunks", {"doc_id": "doc123"})
            # Deletes from physical table with mapped field names
        """
        table = self.get_table(logical_table)
        stmt = delete(table)

        for logical_field, value in filters.items():
            physical_field = self._map_field(logical_field)
            stmt = stmt.where(table.c[physical_field] == value)

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def vector_search(
        self,
        logical_table: str,
        vector_field: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine"
    ) -> List[Tuple[Any, float]]:
        """
        Perform vector similarity search using pgvector.

        Args:
            logical_table: Logical table name
            vector_field: Logical field name containing vectors
            query_vector: Query vector as list of floats
            top_k: Number of results to return
            distance_metric: "cosine", "l2", or "inner_product"

        Returns:
            List of (row, distance) tuples sorted by distance

        Raises:
            ValueError: If distance_metric is invalid

        Example:
            >>> results = context.vector_search(
            ...     "vectors",
            ...     vector_field="embedding",
            ...     query_vector=[0.1, 0.2, ...],  # 384 dimensions
            ...     top_k=5,
            ...     distance_metric="cosine"
            ... )
            >>> for row, distance in results:
            ...     print(f"Distance: {distance}, Content: {row.content}")
        """
        table = self.get_table(logical_table)
        physical_vector_field = self._map_field(vector_field)

        # pgvector distance functions
        if distance_metric == "cosine":
            distance_func = table.c[physical_vector_field].cosine_distance
        elif distance_metric == "l2":
            distance_func = table.c[physical_vector_field].l2_distance
        elif distance_metric == "inner_product":
            distance_func = table.c[physical_vector_field].max_inner_product
        else:
            raise ValueError(
                f"Unknown distance metric: '{distance_metric}'. "
                f"Valid options: 'cosine', 'l2', 'inner_product'"
            )

        query = select(
            table,
            distance_func(query_vector).label('distance')
        ).order_by('distance').limit(top_k)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchall()
```

#### PostgresqlDatabaseService Extension

```python
# rag_factory/services/database/postgres_service.py
"""
PostgreSQL database service with connection pooling and context support.
Extended from Epic 16 to support DatabaseContext.
"""
from typing import Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .database_context import DatabaseContext
from .interfaces import IDatabaseService


class PostgresqlDatabaseService(IDatabaseService):
    """
    PostgreSQL database service with connection pooling.

    Extended to support strategy-specific table contexts that share
    the same connection pool for efficiency.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize PostgreSQL service with connection pooling.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Number of connections in pool
            max_overflow: Max overflow connections beyond pool_size
        """
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True  # Verify connections before using
        )
        self._contexts: Dict[tuple, DatabaseContext] = {}  # Cache contexts

    def get_context(
        self,
        table_mapping: Dict[str, str],
        field_mapping: Optional[Dict[str, str]] = None
    ) -> DatabaseContext:
        """
        Create strategy-specific database context.

        Multiple contexts share the same connection pool (same engine)
        but have different table/field mappings for isolation.

        Args:
            table_mapping: Dict mapping logical → physical table names
                          e.g., {"chunks": "semantic_chunks", "vectors": "semantic_vectors"}
            field_mapping: Optional dict mapping logical → physical field names
                          e.g., {"content": "text_content", "embedding": "vector_embedding"}

        Returns:
            DatabaseContext with specified mappings and shared engine

        Example:
            >>> # Strategy 1: Semantic search
            >>> semantic_ctx = db_service.get_context(
            ...     table_mapping={"chunks": "semantic_chunks", "vectors": "semantic_vectors"},
            ...     field_mapping={"content": "text_content"}
            ... )
            >>>
            >>> # Strategy 2: Keyword search (same DB, different tables)
            >>> keyword_ctx = db_service.get_context(
            ...     table_mapping={"chunks": "keyword_chunks", "index": "keyword_inverted_index"}
            ... )
            >>>
            >>> # Both share same connection pool
            >>> assert semantic_ctx.engine is keyword_ctx.engine  # True
        """
        # Create unique key from mappings for caching
        table_key = frozenset(table_mapping.items())
        field_key = frozenset(field_mapping.items()) if field_mapping else frozenset()
        cache_key = (table_key, field_key)

        # Return cached context if exists
        if cache_key not in self._contexts:
            self._contexts[cache_key] = DatabaseContext(
                engine=self.engine,  # Shared engine
                table_mapping=table_mapping,
                field_mapping=field_mapping
            )

        return self._contexts[cache_key]

    def close(self) -> None:
        """Close database connection pool and all contexts."""
        self._contexts.clear()
        self.engine.dispose()

    # ... other IDatabaseService methods from Epic 16 ...
```

### Usage Examples

#### Example 1: Basic Usage

```python
from rag_factory.services.database import PostgresqlDatabaseService

# Create database service
db_service = PostgresqlDatabaseService(
    connection_string="postgresql://user:pass@localhost:5432/rag_db",
    pool_size=10
)

# Create context for semantic search strategy
semantic_context = db_service.get_context(
    table_mapping={
        "chunks": "semantic_chunks",
        "vectors": "semantic_vectors"
    },
    field_mapping={
        "content": "text_content",
        "embedding": "vector_embedding",
        "doc_id": "document_id"
    }
)

# Use logical names in strategy code
semantic_context.insert("chunks", {
    "content": "Python is a programming language",
    "doc_id": "doc123",
    "chunk_index": 0
})

# Query using logical names
results = semantic_context.query(
    "chunks",
    filters={"doc_id": "doc123"},
    limit=10
)

# Vector search using logical names
matches = semantic_context.vector_search(
    "vectors",
    vector_field="embedding",
    query_vector=[0.1, 0.2, ...],  # 384-dimensional vector
    top_k=5,
    distance_metric="cosine"
)
```

#### Example 2: Multiple Strategies Sharing Database

```python
# Get shared database service (from ServiceRegistry in Epic 17)
db_service = registry.get("db1")

# Strategy 1: Semantic search
semantic_context = db_service.get_context(
    table_mapping={
        "chunks": "semantic_chunks",
        "vectors": "semantic_vectors"
    },
    field_mapping={
        "content": "text_content",
        "embedding": "vector_embedding"
    }
)

# Strategy 2: Keyword search (same DB, different tables)
keyword_context = db_service.get_context(
    table_mapping={
        "chunks": "keyword_chunks",
        "index": "keyword_inverted_index"
    }
)

# Both use same connection pool
assert semantic_context.engine is keyword_context.engine  # True

# But write to different tables
semantic_context.insert("chunks", {"content": "hello", "doc_id": "123"})
# Inserts into semantic_chunks.text_content

keyword_context.insert("chunks", {"content": "hello", "doc_id": "123"})
# Inserts into keyword_chunks.content (different table!)
```

#### Example 3: Integration with Strategy Pair (Epic 17)

```yaml
# strategies/semantic-local-pair.yaml
strategy_name: "semantic-local-pair"
version: "1.0.0"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding1"
    db: "$db1"

  # DatabaseContext configuration
  db_config:
    tables:
      chunks: "semantic_local_chunks"
      vectors: "semantic_local_vectors"
      metadata: "semantic_local_metadata"
    fields:
      content: "text_content"
      embedding: "vector_embedding"
      doc_id: "document_id"
```

```python
# rag_factory/manager/strategy_pair_manager.py (Epic 17)
def _create_strategy(self, config: dict, is_indexing: bool):
    """Instantiate strategy with DatabaseContext"""

    # Get services from registry
    db_service = self.registry.get(config['services']['db'])

    # Create DatabaseContext from db_config
    if 'db_config' in config:
        db_context = db_service.get_context(
            table_mapping=config['db_config'].get('tables', {}),
            field_mapping=config['db_config'].get('fields', {})
        )

        # Pass context to strategy
        strategy = StrategyClass(
            db=db_context,  # DatabaseContext instead of service
            **other_services
        )

    return strategy
```

---

## Testing Strategy

### Unit Tests

#### Test 1: DatabaseContext Initialization

```python
# tests/unit/services/database/test_database_context.py
import pytest
from sqlalchemy import create_engine
from rag_factory.services.database import DatabaseContext


def test_database_context_initialization():
    """Test DatabaseContext initializes with correct parameters"""
    engine = create_engine("sqlite:///:memory:")
    table_mapping = {"chunks": "physical_chunks"}
    field_mapping = {"content": "text_content"}

    context = DatabaseContext(engine, table_mapping, field_mapping)

    assert context.engine is engine
    assert context.tables == table_mapping
    assert context.fields == field_mapping
    assert context._metadata is not None
    assert context._reflected_tables == {}


def test_field_mapping_optional():
    """Test DatabaseContext works without field mapping"""
    engine = create_engine("sqlite:///:memory:")
    table_mapping = {"chunks": "physical_chunks"}

    context = DatabaseContext(engine, table_mapping)

    assert context.fields == {}
    assert context._map_field("content") == "content"  # Pass-through


def test_field_mapping_passthrough():
    """Test unmapped fields pass through unchanged"""
    engine = create_engine("sqlite:///:memory:")
    field_mapping = {"content": "text_content"}

    context = DatabaseContext(engine, {}, field_mapping)

    assert context._map_field("content") == "text_content"
    assert context._map_field("unmapped_field") == "unmapped_field"
```

#### Test 2: Table Name Mapping and Errors

```python
def test_get_table_unmapped_logical_name_raises_error():
    """Test clear error when logical table name not in mapping"""
    engine = create_engine("sqlite:///:memory:")
    table_mapping = {"chunks": "physical_chunks"}

    context = DatabaseContext(engine, table_mapping)

    with pytest.raises(KeyError) as exc_info:
        context.get_table("nonexistent")

    assert "No table mapping for 'nonexistent'" in str(exc_info.value)
    assert "Available logical names: ['chunks']" in str(exc_info.value)
```

#### Test 3: CRUD Operations

```python
# tests/unit/services/database/test_database_context_crud.py
import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from rag_factory.services.database import DatabaseContext


@pytest.fixture
def test_db_with_tables():
    """Create in-memory SQLite DB with test tables"""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Create physical table
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


def test_insert_with_field_mapping(test_db_with_tables):
    """Test insert maps logical fields to physical fields"""
    context = DatabaseContext(
        test_db_with_tables,
        table_mapping={"chunks": "physical_chunks"},
        field_mapping={
            "content": "text_content",
            "doc_id": "document_id"
        }
    )

    # Insert with logical names
    context.insert("chunks", {
        "chunk_id": "chunk1",
        "content": "Hello world",  # Logical name
        "doc_id": "doc123",       # Logical name
        "chunk_index": 0
    })

    # Verify data inserted with physical field names
    results = context.query("chunks", filters={"doc_id": "doc123"})
    assert len(results) == 1
    assert results[0].text_content == "Hello world"  # Physical name
    assert results[0].document_id == "doc123"        # Physical name


def test_query_with_filters(test_db_with_tables):
    """Test query applies filter mapping correctly"""
    context = DatabaseContext(
        test_db_with_tables,
        table_mapping={"chunks": "physical_chunks"},
        field_mapping={"doc_id": "document_id"}
    )

    # Insert test data
    context.insert("chunks", {
        "chunk_id": "chunk1",
        "text_content": "Test",
        "doc_id": "doc123",
        "chunk_index": 0
    })
    context.insert("chunks", {
        "chunk_id": "chunk2",
        "text_content": "Test 2",
        "doc_id": "doc456",
        "chunk_index": 1
    })

    # Query with logical field name
    results = context.query("chunks", filters={"doc_id": "doc123"})

    assert len(results) == 1
    assert results[0].chunk_id == "chunk1"


def test_update_with_mapping(test_db_with_tables):
    """Test update maps both filters and updates"""
    context = DatabaseContext(
        test_db_with_tables,
        table_mapping={"chunks": "physical_chunks"},
        field_mapping={
            "content": "text_content",
            "doc_id": "document_id"
        }
    )

    # Insert
    context.insert("chunks", {
        "chunk_id": "chunk1",
        "content": "Original",
        "doc_id": "doc123",
        "chunk_index": 0
    })

    # Update with logical names
    context.update(
        "chunks",
        filters={"doc_id": "doc123"},
        updates={"content": "Updated"}
    )

    # Verify
    results = context.query("chunks", filters={"doc_id": "doc123"})
    assert results[0].text_content == "Updated"


def test_delete_with_mapping(test_db_with_tables):
    """Test delete maps filter fields correctly"""
    context = DatabaseContext(
        test_db_with_tables,
        table_mapping={"chunks": "physical_chunks"},
        field_mapping={"doc_id": "document_id"}
    )

    # Insert
    context.insert("chunks", {
        "chunk_id": "chunk1",
        "text_content": "Test",
        "doc_id": "doc123",
        "chunk_index": 0
    })

    # Delete with logical name
    context.delete("chunks", filters={"doc_id": "doc123"})

    # Verify deleted
    results = context.query("chunks")
    assert len(results) == 0
```

#### Test 4: Vector Search Operations

```python
# tests/unit/services/database/test_database_context_vector.py
import pytest
from sqlalchemy import create_engine, MetaData, Table, Column, String
from pgvector.sqlalchemy import Vector
from rag_factory.services.database import DatabaseContext


@pytest.fixture
def test_db_with_vectors():
    """Create test DB with pgvector support"""
    # Note: Requires PostgreSQL with pgvector in real tests
    # This is a simplified example
    engine = create_engine("postgresql://localhost/test_db")
    metadata = MetaData()

    Table(
        'physical_vectors',
        metadata,
        Column('id', String, primary_key=True),
        Column('vector_embedding', Vector(384)),
        Column('text_content', String)
    )

    metadata.create_all(engine)
    return engine


def test_vector_search_cosine_distance(test_db_with_vectors):
    """Test vector search with cosine distance"""
    context = DatabaseContext(
        test_db_with_vectors,
        table_mapping={"vectors": "physical_vectors"},
        field_mapping={"embedding": "vector_embedding"}
    )

    # Insert test vectors
    test_vector1 = [0.1] * 384
    test_vector2 = [0.9] * 384

    context.insert("vectors", {
        "id": "vec1",
        "embedding": test_vector1,
        "text_content": "Hello"
    })
    context.insert("vectors", {
        "id": "vec2",
        "embedding": test_vector2,
        "text_content": "World"
    })

    # Search with logical names
    query_vector = [0.15] * 384  # Closer to test_vector1
    results = context.vector_search(
        "vectors",
        vector_field="embedding",
        query_vector=query_vector,
        top_k=2,
        distance_metric="cosine"
    )

    assert len(results) == 2
    # First result should be closer (smaller distance)
    assert results[0].distance < results[1].distance
    assert results[0].text_content == "Hello"


def test_vector_search_invalid_metric_raises_error(test_db_with_vectors):
    """Test clear error for invalid distance metric"""
    context = DatabaseContext(
        test_db_with_vectors,
        table_mapping={"vectors": "physical_vectors"}
    )

    with pytest.raises(ValueError) as exc_info:
        context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=[0.1] * 384,
            distance_metric="invalid_metric"
        )

    assert "Unknown distance metric: 'invalid_metric'" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)
```

### Integration Tests

#### Test 5: Multi-Context Isolation

```python
# tests/integration/database/test_multi_context_isolation.py
import pytest
from rag_factory.services.database import PostgresqlDatabaseService


@pytest.fixture
def postgres_service():
    """Create PostgreSQL service for testing"""
    return PostgresqlDatabaseService(
        connection_string="postgresql://localhost/test_rag_db",
        pool_size=5
    )


def test_multiple_contexts_share_engine(postgres_service):
    """Test multiple contexts share same connection pool"""
    # Create two contexts
    context1 = postgres_service.get_context(
        table_mapping={"chunks": "semantic_chunks"}
    )
    context2 = postgres_service.get_context(
        table_mapping={"chunks": "keyword_chunks"}
    )

    # Verify they share the same engine
    assert context1.engine is context2.engine
    assert context1.engine is postgres_service.engine


def test_contexts_isolated_different_tables(postgres_service):
    """Test contexts write to different physical tables"""
    # Context for semantic search
    semantic_ctx = postgres_service.get_context(
        table_mapping={"chunks": "semantic_chunks"},
        field_mapping={"content": "text_content"}
    )

    # Context for keyword search
    keyword_ctx = postgres_service.get_context(
        table_mapping={"chunks": "keyword_chunks"}
    )

    # Both insert to logical "chunks" table
    semantic_ctx.insert("chunks", {
        "chunk_id": "s1",
        "content": "Semantic content",
        "doc_id": "doc1"
    })

    keyword_ctx.insert("chunks", {
        "chunk_id": "k1",
        "content": "Keyword content",
        "doc_id": "doc1"
    })

    # Verify isolation - each context only sees its own data
    semantic_results = semantic_ctx.query("chunks")
    keyword_results = keyword_ctx.query("chunks")

    assert len(semantic_results) == 1
    assert len(keyword_results) == 1
    assert semantic_results[0].chunk_id == "s1"
    assert keyword_results[0].chunk_id == "k1"


def test_context_caching(postgres_service):
    """Test contexts are cached for same mappings"""
    table_mapping = {"chunks": "test_chunks"}
    field_mapping = {"content": "text"}

    # Get context twice with same mappings
    context1 = postgres_service.get_context(table_mapping, field_mapping)
    context2 = postgres_service.get_context(table_mapping, field_mapping)

    # Should return same instance
    assert context1 is context2

    # Different mapping should create new context
    context3 = postgres_service.get_context({"chunks": "other_chunks"})
    assert context3 is not context1
```

### Performance Tests

```python
# tests/performance/test_database_context_performance.py
import time
import pytest
from rag_factory.services.database import PostgresqlDatabaseService


def test_context_creation_performance(postgres_service):
    """Test context creation is fast"""
    start = time.time()

    for i in range(100):
        context = postgres_service.get_context(
            table_mapping={"chunks": f"table_{i}"}
        )

    elapsed = time.time() - start
    avg_time = elapsed / 100

    assert avg_time < 0.010  # < 10ms per context (most cached)


def test_query_overhead_minimal(postgres_service):
    """Test logical→physical mapping adds minimal overhead"""
    context = postgres_service.get_context(
        table_mapping={"chunks": "test_chunks"},
        field_mapping={"content": "text", "doc_id": "document_id"}
    )

    # Warm up
    context.query("chunks", limit=1)

    # Measure query time
    start = time.time()
    for _ in range(1000):
        context.query("chunks", filters={"doc_id": "doc1"}, limit=10)
    elapsed = time.time() - start

    avg_query_time = (elapsed / 1000) * 1000  # Convert to ms

    assert avg_query_time < 5  # < 5ms average (including DB time)
```

---

## Implementation Steps

### Phase 1: Core DatabaseContext (Days 1-2)
1. Create `database_context.py` file
2. Implement `DatabaseContext.__init__`
3. Implement `get_table()` with reflection and caching
4. Implement `_map_field()` helper
5. Write unit tests for initialization and mapping

### Phase 2: CRUD Operations (Days 3-4)
1. Implement `insert()` method
2. Implement `query()` method
3. Implement `update()` method
4. Implement `delete()` method
5. Write comprehensive unit tests for each operation

### Phase 3: Vector Search (Day 5)
1. Implement `vector_search()` method
2. Add support for cosine, L2, inner product metrics
3. Write unit tests for vector operations
4. Test with real pgvector extension

### Phase 4: Service Integration (Day 6)
1. Extend `PostgresqlDatabaseService` with `get_context()`
2. Implement context caching
3. Write integration tests
4. Verify connection pool sharing

### Phase 5: Testing & Documentation (Days 7-8)
1. Write integration tests for multi-context isolation
2. Write performance benchmarks
3. Complete docstrings
4. Write usage examples
5. Create migration guide

---

## Migration Guide

### For Existing Code Using PostgresqlDatabaseService

**Before (Epic 16):**
```python
db_service = PostgresqlDatabaseService("postgresql://...")
db_service.execute("INSERT INTO chunks VALUES ...")
```

**After (Epic 17):**
```python
db_service = PostgresqlDatabaseService("postgresql://...")
context = db_service.get_context(
    table_mapping={"chunks": "semantic_chunks"}
)
context.insert("chunks", {...})
```

### For Strategy Implementations

**Before:**
```python
class MyStrategy:
    def __init__(self, db_service: PostgresqlDatabaseService):
        self.db = db_service

    def index(self, documents):
        # Hard-coded table name
        self.db.execute("INSERT INTO chunks ...")
```

**After:**
```python
class MyStrategy:
    def __init__(self, db: DatabaseContext):
        self.db = db  # Now receives context instead of service

    def index(self, documents):
        # Logical table name (mapped to physical)
        self.db.insert("chunks", {...})
```

---

## Success Metrics

1. **Isolation:** Multiple strategies can coexist without interference
2. **Performance:** No measurable overhead for logical→physical mapping
3. **Resource Efficiency:** Single connection pool shared by all contexts
4. **Developer Experience:** Clear APIs and helpful error messages
5. **Test Coverage:** >90% coverage for DatabaseContext
6. **Integration:** Seamless integration with Epic 17 ServiceRegistry

---

## Documentation Deliverables

1. **API Documentation:**
   - Complete docstrings for all public methods
   - Usage examples in docstrings
   - Parameter descriptions and types

2. **Integration Guide:**
   - How to use DatabaseContext with strategies
   - How to configure table/field mappings
   - How to integrate with ServiceRegistry (Epic 17)

3. **Performance Guide:**
   - Caching behavior
   - Connection pool usage
   - Best practices for mappings

4. **Migration Guide:**
   - Upgrading from Epic 16 direct DB access
   - Converting strategies to use DatabaseContext
   - Testing strategy isolation

5. **Troubleshooting:**
   - Common errors and solutions
   - Debugging context mappings
   - Performance tuning tips

---

## Related Stories

- **Story 17.2:** Service Registry (provides DB service instance)
- **Story 17.4:** Migration Validator (validates DB schema)
- **Story 17.5:** StrategyPairManager (creates contexts from config)
- **Epic 11:** Dependency Injection (service interfaces)
- **Epic 16:** Database Consolidation (PostgresqlDatabaseService base)

---

## References

- SQLAlchemy MetaData Reflection: https://docs.sqlalchemy.org/en/20/core/reflection.html
- pgvector Extension: https://github.com/pgvector/pgvector
- Connection Pooling: https://docs.sqlalchemy.org/en/20/core/pooling.html
- PostgreSQL Table Partitioning: https://www.postgresql.org/docs/current/ddl-partitioning.html

---

## Notes

- DatabaseContext is read-heavy (queries) not write-heavy, so caching is very effective
- Table reflection happens once per context and is cached
- Connection pool is shared across all contexts for maximum efficiency
- Each strategy pair configuration will specify its own table/field mappings
- This design enables multiple RAG strategies to coexist on one database
