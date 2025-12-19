"""PostgreSQL database service implementation with pgvector.

This module provides a database service that implements IDatabaseService
using PostgreSQL with pgvector extension for vector similarity search.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging
import json

from rag_factory.services.database.database_context import DatabaseContext

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

if TYPE_CHECKING:
    import asyncpg

try:
    import asyncpg  # noqa: F811
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from rag_factory.services.interfaces import IDatabaseService

logger = logging.getLogger(__name__)


class PostgresqlDatabaseService(IDatabaseService):
    """PostgreSQL database service with pgvector.

    This service implements IDatabaseService using PostgreSQL with the
    pgvector extension for storing and retrieving document chunks with
    vector embeddings.

    Example:
        >>> service = PostgresqlDatabaseService(
        ...     host="localhost",
        ...     port=5432,
        ...     database="rag_db",
        ...     user="postgres",
        ...     password="password"
        ... )
        >>> chunks = [{
        ...     "text": "Hello world",
        ...     "embedding": [0.1, 0.2, 0.3],
        ...     "metadata": {"source": "doc1"}
        ... }]
        >>> await service.store_chunks(chunks)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "rag_db",
        user: str = "postgres",
        password: str = "",
        table_name: str = "chunks",
        vector_dimensions: int = 384,
        connection_string: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """Initialize PostgreSQL database service.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table_name: Name of the chunks table (default: chunks)
            vector_dimensions: Dimension of embedding vectors (default: 384)

        Raises:
            ImportError: If asyncpg package is not installed
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg package not installed. Install with:\n"
                "  pip install asyncpg"
            )

        if connection_string:
            try:
                from sqlalchemy.engine import make_url
                url = make_url(connection_string)
                self.host = url.host or "localhost"
                self.port = url.port or 5432
                self.database = url.database or "rag_db"
                self.user = url.username or "postgres"
                self.password = url.password or ""
            except ImportError:
                 # Fallback if sqlalchemy not available? But file imports it.
                 # Assuming simple parsing if make_url fails or not imported?
                 # Since imports are usually at top and handled, it should be fine.
                 raise ImportError("SQLAlchemy required to parse connection string")
        else:
            self.host = host
            self.port = port
            self.database = database
            self.user = user
            self.password = password
            
        self.table_name = table_name
        self.vector_dimensions = vector_dimensions
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._pool: Optional["asyncpg.Pool"] = None  # type: ignore
        
        # Synchronous engine for DatabaseContext (Epic 17)
        self._sync_engine: Optional["Engine"] = None  # type: ignore
        self._contexts: Dict[tuple, DatabaseContext] = {}  # Cache contexts

        logger.info(
            f"Initialized PostgreSQL service for {host}:{port}/{database}"
        )

    async def _get_pool(self) -> "asyncpg.Pool":  # type: ignore
        """Get or create connection pool.

        Returns:
            asyncpg connection pool
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=self.pool_size
            )

            # Ensure table exists
            await self._ensure_table()

        return self._pool

    async def _ensure_table(self):
        """Ensure the chunks table exists with pgvector extension."""
        logger.info(f"[_ensure_table] Starting table setup for {self.table_name}")
        logger.info(f"[_ensure_table] Acquiring connection from pool")
        async with self._pool.acquire() as conn:
            logger.info(f"[_ensure_table] Connection acquired, enabling pgvector extension")
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info(f"[_ensure_table] pgvector extension enabled")

            # Create chunks table
            logger.info(f"[_ensure_table] Creating table {self.table_name}")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    text TEXT NOT NULL,
                    embedding vector({self.vector_dimensions}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info(f"[_ensure_table] Table created/verified")

            # Check if embedding column exists (validation for existing tables)
            logger.info(f"[_ensure_table] Checking if embedding column exists")
            column_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns 
                    WHERE table_name = $1 
                    AND column_name = 'embedding'
                )
                """,
                self.table_name
            )
            logger.info(f"[_ensure_table] Embedding column exists: {column_exists}")

            if not column_exists:
                logger.info(f"Adding missing 'embedding' column to table {self.table_name}")
                await conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN embedding vector({self.vector_dimensions})")
                logger.info(f"[_ensure_table] Embedding column added")

            # Create index for vector similarity search
            logger.info(f"[_ensure_table] Creating ivfflat index")
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            logger.info(f"[_ensure_table] Index created/verified")

            logger.info(f"Ensured table {self.table_name} exists")

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store document chunks.

        Args:
            chunks: List of chunk dictionaries. Each chunk should contain
                   at minimum: text content and embedding vector. Additional
                   fields like metadata, chunk_id, etc. are implementation-specific.

        Raises:
            Exception: If storage fails
        """
        if not chunks:
            return

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Prepare data for insertion
            for chunk in chunks:
                # Handle Chunk objects (dataclass or SQLAlchemy ORM) and dictionaries
                if isinstance(chunk, dict):
                    # It's a dictionary
                    chunk_id = chunk.get("chunk_id", chunk.get("id"))
                    text = chunk.get("text", "")
                    embedding = chunk.get("embedding", [])
                    metadata = chunk.get("metadata", {})
                else:
                    # It's a Chunk object (dataclass or ORM)
                    chunk_id = getattr(chunk, 'chunk_id', getattr(chunk, 'id', None))
                    text = getattr(chunk, 'text', '')
                    embedding = getattr(chunk, 'embedding', [])
                    # Handle both 'metadata' and 'metadata_' attributes (ORM uses metadata_)
                    metadata = getattr(chunk, 'metadata', getattr(chunk, 'metadata_', {}))

                # Convert embedding to list if it's a numpy array or similar
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding) if embedding else []

                # Skip chunks without embeddings
                if not embedding:
                    logger.warning(f"Skipping chunk {chunk_id} - no embedding provided")
                    continue

                # Convert embedding to string format for pgvector
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                
                logger.debug(f"Storing chunk {chunk_id}: embedding dims={len(embedding)}, first 3 values={embedding[:3]}")

                # Insert or update chunk
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (chunk_id, text, embedding, metadata)
                    VALUES ($1, $2, '{embedding_str}'::vector, $3)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    chunk_id,
                    text,
                    json.dumps(metadata)
                )

        logger.debug(f"Stored {len(chunks)} chunks")

    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chunks by similarity.

        Performs vector similarity search to find chunks most similar
        to the query embedding.

        Args:
            query_embedding: Query vector to search for
            top_k: Maximum number of results to return

        Returns:
            List of Chunk objects, sorted by similarity score in
            descending order.

        Raises:
            Exception: If search fails
        """
        from rag_factory.repositories.chunk import Chunk
        pool = await self._get_pool()

        # Convert embedding to string format for pgvector
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        logger.debug(f"Searching with embedding dims={len(query_embedding)}, first 3 values={query_embedding[:3]}")

        async with pool.acquire() as conn:
            # Set probes for better recall with IVFFlat index
            # With lists=100 (set in _ensure_table), probes=10 searches 10% of clusters
            # This significantly improves recall while maintaining good query performance
            await conn.execute("SET ivfflat.probes = 10")
            
            # Debug: count chunks with embeddings
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name} WHERE embedding IS NOT NULL")
            logger.debug(f"Found {count} chunks with embeddings in database")
            
            try:
                # Note: We embed the vector string directly in the SQL because
                # pgvector may not properly handle parameterized vector values
                rows = await conn.fetch(
                    f"""
                    SELECT
                        chunk_id,
                        text,
                        metadata,
                        1 - (embedding <=> '{embedding_str}'::vector) as similarity
                    FROM {self.table_name}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> '{embedding_str}'::vector
                    LIMIT $1
                    """,
                    top_k
                )
            except Exception as e:
                logger.error(f"Vector search query failed: {e}")
                logger.error(f"Embedding string length: {len(embedding_str)}")
                raise

        # Convert rows to Chunk objects
        results = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                embedding=[]  # Don't return embeddings in search results to save memory
            )
            # Add similarity as metadata for convenience
            chunk.metadata['similarity'] = float(row["similarity"])
            results.append(chunk)

        logger.debug(f"Found {len(results)} similar chunks")
        return results

    async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve chunk by ID.

        Args:
            chunk_id: Unique identifier of the chunk to retrieve

        Returns:
            Chunk dictionary containing text content and metadata

        Raises:
            Exception: If chunk is not found or retrieval fails
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT chunk_id, text, metadata
                FROM {self.table_name}
                WHERE chunk_id = $1
                """,
                chunk_id
            )

        if not row:
            raise ValueError(f"Chunk not found: {chunk_id}")

        return {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }

    async def get_all_chunks(self) -> List[Any]:
        """Retrieve all chunks from the database.

        Returns:
            List of Chunk objects

        Raises:
            Exception: If retrieval fails
        """
        from rag_factory.repositories.chunk import Chunk
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT chunk_id, text, metadata, embedding
                FROM {self.table_name}
                ORDER BY created_at
                """
            )

        # Convert rows to Chunk objects
        chunks = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                embedding=list(row["embedding"]) if row["embedding"] else []
            )
            chunks.append(chunk)

        logger.debug(f"Retrieved {len(chunks)} chunks")
        return chunks

    async def get_chunks_for_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a list of documents.

        Args:
            document_ids: List of document IDs to retrieve chunks for

        Returns:
            List of chunk dictionaries belonging to the specified documents
        """
        if not document_ids:
            return []

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT chunk_id, text, metadata
                FROM {self.table_name}
                WHERE metadata->>'document_id' = ANY($1)
                """,
                document_ids
            )

        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            for row in rows
        ]

    async def store_chunks_with_hierarchy(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks with hierarchical metadata.
        
        This method stores chunks that include hierarchical relationship metadata
        such as parent-child relationships, hierarchy levels, and path information.
        
        Args:
            chunks: List of chunk dictionaries with hierarchy metadata.
                   Each chunk should contain:
                   - id: Unique chunk identifier
                   - document_id: ID of the parent document
                   - text: Chunk text content
                   - level: Hierarchy level (0 = document, 1 = section, 2 = paragraph)
                   - parent_id: ID of parent chunk (None for root level)
                   - path: List of indices representing path from root
                   - metadata: Additional metadata dictionary
        
        Raises:
            Exception: If storage fails
        """
        # For PostgreSQL with pgvector, we store the hierarchy info in metadata
        # and the text/embedding as usual.
        
        enriched_chunks = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {}).copy()
            
            # Add hierarchy fields to metadata
            hierarchy_fields = ["level", "parent_id", "path", "document_id"]
            for field in hierarchy_fields:
                if field in chunk:
                    metadata[field] = chunk[field]
            
            enriched_chunk = {
                "chunk_id": chunk.get("id"),
                "text": chunk.get("text"),
                "embedding": chunk.get("embedding"),
                "metadata": metadata
            }
            enriched_chunks.append(enriched_chunk)
            
        await self.store_chunks(enriched_chunks)

    def _get_sync_engine(self) -> "Engine":  # type: ignore
        """Get or create synchronous SQLAlchemy engine.

        This engine is used for DatabaseContext instances and uses
        connection pooling for efficiency.

        Returns:
            SQLAlchemy Engine instance

        Raises:
            ImportError: If SQLAlchemy is not installed
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy package not installed. Install with:\n"
                "  pip install sqlalchemy psycopg2-binary"
            )

        if self._sync_engine is None:
            # Build connection string for SQLAlchemy
            password_part = f":{self.password}" if self.password else ""
            connection_string = (
                f"postgresql://{self.user}{password_part}@"
                f"{self.host}:{self.port}/{self.database}"
            )

            self._sync_engine = create_engine(
                connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True  # Verify connections before using
            )

            logger.info("Created synchronous SQLAlchemy engine for contexts")

        return self._sync_engine

    @property
    def engine(self) -> "Engine":
        """Access the synchronous SQLAlchemy engine."""
        return self._get_sync_engine()

    @engine.setter
    def engine(self, value: "Engine") -> None:
        """Set the synchronous SQLAlchemy engine.
        
        This is primarily used for testing to inject a test database engine.
        
        Args:
            value: SQLAlchemy Engine instance to use
        """
        self._sync_engine = value

    def get_context(
        self,
        table_mapping: Dict[str, str],
        field_mapping: Optional[Dict[str, str]] = None
    ) -> DatabaseContext:
        """Create strategy-specific database context.

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
            engine = self._get_sync_engine()
            self._contexts[cache_key] = DatabaseContext(
                engine=engine,  # Shared engine
                table_mapping=table_mapping,
                field_mapping=field_mapping
            )
            logger.debug(
                f"Created new DatabaseContext with {len(table_mapping)} table mappings"
            )
        else:
            logger.debug("Returning cached DatabaseContext")

        return self._contexts[cache_key]


    async def close(self):
        """Close the database connection pool.

        Should be called when the service is no longer needed.
        Also disposes of synchronous engine and clears context cache.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL async connection pool")
        
        # Close synchronous engine and clear contexts
        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            logger.info("Disposed synchronous SQLAlchemy engine")
        
        self._contexts.clear()
        logger.debug("Cleared DatabaseContext cache")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
