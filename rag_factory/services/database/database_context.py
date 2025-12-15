"""Database context providing strategy-specific view of shared database.

This module implements DatabaseContext, which provides a strategy-specific
view of a shared database through table and field name mapping. This enables
multiple RAG strategies to coexist on the same database without interference.
"""

from typing import Dict, Optional, List, Any, Tuple
from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)


class DatabaseContext:
    """Strategy-specific view of a shared database.

    Provides access to tables using logical names that map to
    physical table names, enabling strategy isolation on shared database.

    The context wraps a SQLAlchemy engine and provides CRUD operations
    with automatic translation of logical table/field names to physical
    names in the database.

    Example:
        >>> db_service = registry.get("db1")
        >>> context = db_service.get_context(
        ...     table_mapping={"chunks": "semantic_chunks"},
        ...     field_mapping={"content": "text_content"}
        ... )
        >>> context.insert("chunks", {"content": "hello", "doc_id": "123"})
        # Inserts into semantic_chunks.text_content

    Attributes:
        engine: SQLAlchemy engine (shared across contexts)
        tables: Dict mapping logical → physical table names
        fields: Dict mapping logical → physical field names
    """

    def __init__(
        self,
        engine: Engine,
        table_mapping: Dict[str, str],
        field_mapping: Optional[Dict[str, str]] = None
    ):
        """Create database context with table/field mappings.

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

        logger.debug(
            f"Created DatabaseContext with {len(table_mapping)} table mappings "
            f"and {len(self.fields)} field mappings"
        )

    def get_table(self, logical_name: str) -> Table:
        """Get SQLAlchemy Table object by logical name.

        Reflects the table structure from the database on first access
        and caches it for subsequent calls.

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
            logger.debug(f"Reflecting table '{physical_name}' from database")
            self._reflected_tables[physical_name] = Table(
                physical_name,
                self._metadata,
                autoload_with=self.engine
            )

        return self._reflected_tables[physical_name]

    def _map_field(self, logical_field: str) -> str:
        """Map logical field name to physical field name.

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
        """Insert row into a logically-named table.

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

        logger.debug(
            f"Inserted row into '{logical_table}' "
            f"(physical: '{self.tables[logical_table]}')"
        )

    def query(
        self,
        logical_table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """Query a logically-named table.

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
            rows = result.fetchall()

        logger.debug(
            f"Queried '{logical_table}' "
            f"(physical: '{self.tables[logical_table]}'), "
            f"returned {len(rows)} rows"
        )
        return rows

    def update(
        self,
        logical_table: str,
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> None:
        """Update rows in a logically-named table.

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

        logger.debug(
            f"Updated rows in '{logical_table}' "
            f"(physical: '{self.tables[logical_table]}')"
        )

    def delete(self, logical_table: str, filters: Dict[str, Any]) -> None:
        """Delete rows from a logically-named table.

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

        logger.debug(
            f"Deleted rows from '{logical_table}' "
            f"(physical: '{self.tables[logical_table]}')"
        )

    def vector_search(
        self,
        logical_table: str,
        vector_field: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine"
    ) -> List[Tuple[Any, float]]:
        """Perform vector similarity search using pgvector.

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
            rows = result.fetchall()

        logger.debug(
            f"Vector search on '{logical_table}' "
            f"(physical: '{self.tables[logical_table]}'), "
            f"returned {len(rows)} results"
        )
        return rows
