"""Database context providing strategy-specific view of shared database.

This module implements DatabaseContext, which provides a strategy-specific
view of a shared database through table and field name mapping. This enables
multiple RAG strategies to coexist on the same database without interference.
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
import uuid

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

        return rows

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Async compatibility method for IDatabaseService.store_chunks.
        
        Maps to synchronous insert/upsert on the 'chunks' logical table.
        """
        # Run sync operation in thread wrapper for async compatibility
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_chunks_sync, chunks)

    def _store_chunks_sync(self, chunks: List[Dict[str, Any]]) -> None:
        """Synchronous implementation of store_chunks using the ChunkRepository."""
        if not chunks:
            return
        
        try:
            # Convert document_id to UUID if it's a string
            for chunk in chunks:
                if isinstance(chunk.get("document_id"), str):
                    chunk["document_id"] = uuid.UUID(chunk["document_id"])

            self.chunk_repository.bulk_create(chunks)
            logger.debug(f"Successfully stored {len(chunks)} chunks via ChunkRepository.")
        except Exception as e:
            logger.error(f"Failed to store chunks via repository: {e}", exc_info=True)
            # The repository handles its own rollback, but we re-raise
            raise

    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Async compatibility method for IDatabaseService.search_chunks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._search_chunks_sync, 
            query_embedding, 
            top_k
        )

    def _search_chunks_sync(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Synchronous implementation of search_chunks."""
        
        # Decide which table to search
        search_table = "vectors" if "vectors" in self.tables else "chunks"
        vector_field_logical = "embedding"
        
        # 1. Perform vector search
        # Note: This returns fields from the search_table.
        # If search_table is 'vectors', we get chunk_id, embedding, etc. but NO text.
        results = self.vector_search(
            logical_table=search_table,
            vector_field=vector_field_logical,
            query_vector=query_embedding,
            top_k=top_k
        )
        
        # 2. If text is missing, fetch it from 'chunks' table
        mapped_results = []
        chunk_ids_to_fetch = []
        
        chunk_field_mapped = self._map_field("chunk_id")
        text_field_mapped = self._map_field("text")
        meta_field_mapped = self._map_field("metadata")
        
        # Preliminary map
        for row in results:
            # row is a Row object. Access distance by label.
            distance = getattr(row, 'distance', 0.0) # Safety default?
            
            # Check if row has text
            # We assume row is accessible by column name (physical)
            text_val = None
            if hasattr(row, text_field_mapped):
                text_val = getattr(row, text_field_mapped)
            
            chunk_id = getattr(row, chunk_field_mapped, None)
            
            res_dict = {
                "chunk_id": chunk_id,
                "text": text_val,
                "metadata": getattr(row, meta_field_mapped, {}) if hasattr(row, meta_field_mapped) else {},
                "score": 1 - distance,
                "similarity": 1 - distance
            }
            mapped_results.append(res_dict)
            
            if text_val is None and chunk_id is not None and "chunks" in self.tables:
                chunk_ids_to_fetch.append(chunk_id)

        # 3. Fetch missing text
        if chunk_ids_to_fetch:
            chunks_table = self.get_table("chunks")
            # We need to map chunk_id logical->physical
            # Assume chunk_id field name in chunks table is same as mapped for vectors (global field mapping)
            
            phys_chunk_id = self._map_field("chunk_id")
            phys_text = self._map_field("text")
            phys_meta = self._map_field("metadata")
            
            stmt = select(chunks_table).where(
                chunks_table.c[phys_chunk_id].in_(chunk_ids_to_fetch)
            )
            
            with self.engine.connect() as conn:
                text_rows = conn.execute(stmt).fetchall()
                
            # Create lookup
            text_lookup = {}
            for r in text_rows:
                cid = getattr(r, phys_chunk_id)
                txt = getattr(r, phys_text)
                meta = getattr(r, phys_meta, {})
                text_lookup[cid] = (txt, meta)
                
            # Merge
            for res in mapped_results:
                cid = res["chunk_id"]
                if cid in text_lookup:
                    if not res["text"]:
                        res["text"] = text_lookup[cid][0]
                    if not res["metadata"]:
                         res["metadata"] = text_lookup[cid][1]

        return mapped_results

    async def get_chunks_for_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Async compatibility method for IDatabaseService.get_chunks_for_documents."""
        # Implementation left simple for now or raises generic error if needed.
        # But VectorEmbeddingIndexing uses it!
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_chunks_for_docs_sync, document_ids)

    def _get_chunks_for_docs_sync(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Synchronous implementation."""
        if not document_ids:
            return []
            
        table = self.get_table("chunks")
        meta_col = self._map_field("metadata")
        
        # Filter by metadata->>'document_id'
        # SQLAlchemy JSON access
        # table.c[meta_col]['document_id'].astext.in_(document_ids)
        
        stmt = select(table).where(
            table.c[meta_col]['document_id'].astext.in_(document_ids)
        )
        
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
            
        results = []
        for row in rows:
            results.append({
                "chunk_id": getattr(row, self._map_field("chunk_id")),
                "text": getattr(row, self._map_field("text")),
                "embedding": getattr(row, self._map_field("embedding"), None), # Might be slow if large
                "metadata": getattr(row, self._map_field("metadata"))
            })
        return results

    async def store_keyword_index(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Store keyword inverted index to database.
        
        Args:
            inverted_index: Dict mapping keywords to list of {chunk_id, score} dicts
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_keyword_index_sync, inverted_index)
    
    def _store_keyword_index_sync(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Synchronous implementation of store_keyword_index."""
        if not inverted_index:
            return
        
        # Check if inverted_index table exists
        if "inverted_index" not in self.tables:
            logger.warning("No 'inverted_index' table mapping found, skipping keyword index storage")
            return
        
        table = self.get_table("inverted_index")
        
        with self.engine.begin() as conn:
            # Clear existing index (simple approach)
            conn.execute(delete(table))
            
            # Insert new index entries
            for keyword, chunk_list in inverted_index.items():
                for entry in chunk_list:
                    data = {
                        self._map_field("term"): keyword,
                        self._map_field("chunk_id"): entry['chunk_id'],
                        self._map_field("score"): entry.get('score', 1.0)
                    }
                    conn.execute(insert(table).values(**data))

    async def store_keyword_index(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Store keyword inverted index to database."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_keyword_index_sync, inverted_index)
    
    def _store_keyword_index_sync(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Synchronous implementation of store_keyword_index."""
        if not inverted_index:
            return
        if "inverted_index" not in self.tables:
            logger.warning("No 'inverted_index' table mapping found")
            return
        table = self.get_table("inverted_index")
        try:
            with self.engine.begin() as conn:
                conn.execute(delete(table))
                for keyword, chunk_list in inverted_index.items():
                    for entry in chunk_list:
                        data = {
                            self._map_field("term"): keyword,
                            self._map_field("chunk_id"): entry['chunk_id'],
                            self._map_field("score"): entry.get('score', 1.0)
                        }
                        conn.execute(insert(table).values(**data))
                logger.info(f"Stored keyword index with {len(inverted_index)} terms")
        except Exception as e:
            logger.error(f"Failed to store keyword index: {e}")
            raise
    
    async def search_keyword(self, query_terms: List[str], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search using keyword index."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_keyword_sync, query_terms, top_k)
    
    def _search_keyword_sync(self, query_terms: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Synchronous keyword search."""
        try:
            index_table = self.get_table('inverted_index')
            chunks_table = self.get_table('chunks')
            with self.engine.connect() as conn:
                matching_chunks = {}
                for term in query_terms:
                    query = select(index_table).where(index_table.c.term == term)
                    for row in conn.execute(query).fetchall():
                        chunk_id = row.chunk_id
                        matching_chunks[chunk_id] = matching_chunks.get(chunk_id, 0) + 1
                if not matching_chunks:
                    return []
                results = []
                for chunk_id, term_count in matching_chunks.items():
                    chunk_id_field = self._map_field('chunk_id')
                    text_field = self._map_field('text')
                    doc_id_field = self._map_field('document_id')
                    query = select(chunks_table).where(chunks_table.c[chunk_id_field] == chunk_id)
                    chunk_row = conn.execute(query).fetchone()
                    if chunk_row:
                        results.append({
                            'chunk_id': chunk_id,
                            'text': getattr(chunk_row, text_field, ''),
                            'document_id': getattr(chunk_row, doc_id_field, ''),
                            'score': term_count / len(query_terms),
                            'metadata': {}
                        })
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:top_k]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    @property
    def chunk_repository(self):
        """Get or create ChunkRepository instance with strategy-specific table and fields."""
        if not hasattr(self, '_chunk_repository') or self._chunk_repository is None:
            from rag_factory.repositories.chunk import ChunkRepository
            from sqlalchemy.orm import Session
            
            if not hasattr(self, '_session') or self._session is None:
                self._session = Session(bind=self.engine)
            
            # Get table name from mapping, default to 'chunks'
            chunk_table = self.tables.get('chunks', 'chunks')
            # Pass field mappings to repository
            self._chunk_repository = ChunkRepository(
                self._session, 
                table_name=chunk_table,
                field_mapping=self.fields
            )
            logger.debug(f"Created ChunkRepository in DatabaseContext with table '{chunk_table}' and {len(self.fields)} field mappings")
        
        return self._chunk_repository
    
    @property
    def document_repository(self):
        """Get or create DocumentRepository instance with strategy-specific table."""
        if not hasattr(self, '_document_repository') or self._document_repository is None:
            from rag_factory.repositories.document import DocumentRepository
            from sqlalchemy.orm import Session
            
            if not hasattr(self, '_session') or self._session is None:
                self._session = Session(bind=self.engine)
            
            # Get table name from mapping, default to 'documents'
            doc_table = self.tables.get('documents', 'documents')
            self._document_repository = DocumentRepository(self._session, table_name=doc_table)
            logger.debug(f"Created DocumentRepository in DatabaseContext with table '{doc_table}'")
        
        return self._document_repository
