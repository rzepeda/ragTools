"""Repository for Chunk entity operations with vector search.

This module implements the repository pattern for Chunk entities,
providing CRUD operations and advanced vector similarity search using pgvector.
"""

from typing import Optional, List, Tuple, Dict, Any
from uuid import UUID
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError

from .base import BaseRepository
from .exceptions import (
    EntityNotFoundError,
    DatabaseConnectionError,
    InvalidQueryError
)
from ..database.models import Chunk


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk CRUD operations with vector search.

    This repository provides standard CRUD operations as well as
    specialized vector similarity search methods using cosine distance.
    """

    def get_by_id(self, chunk_id: UUID) -> Optional[Chunk]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: UUID of the chunk to retrieve

        Returns:
            Chunk if found, None otherwise

        Raises:
            DatabaseConnectionError: If database query fails
        """
        try:
            return self.session.query(Chunk).filter(
                Chunk.chunk_id == chunk_id
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query chunk: {str(e)}")

    def get_by_document(
        self,
        document_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Chunk]:
        """Retrieve all chunks for a specific document.

        Chunks are returned ordered by their chunk_index.

        Args:
            document_id: UUID of the parent document
            skip: Number of chunks to skip (for pagination)
            limit: Maximum number of chunks to return

        Returns:
            List of Chunk entities ordered by chunk_index

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            return self.session.query(Chunk)\
                .filter(Chunk.document_id == document_id)\
                .order_by(Chunk.chunk_index)\
                .offset(skip)\
                .limit(limit)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query chunks: {str(e)}")

    def count_by_document(self, document_id: UUID) -> int:
        """Count the number of chunks for a document.

        Args:
            document_id: UUID of the parent document

        Returns:
            Number of chunks for the specified document

        Raises:
            DatabaseConnectionError: If count query fails
        """
        try:
            return self.session.query(func.count(Chunk.chunk_id))\
                .filter(Chunk.document_id == document_id)\
                .scalar()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to count chunks: {str(e)}")

    def create(
        self,
        document_id: UUID,
        chunk_index: int,
        text: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a new chunk.

        Args:
            document_id: UUID of the parent document
            chunk_index: Order index within the document (0-indexed)
            text: The text content of the chunk
            embedding: Optional vector embedding (list of floats)
            metadata: Optional dictionary of custom metadata

        Returns:
            The created Chunk entity

        Raises:
            DatabaseConnectionError: If creation fails
        """
        try:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                embedding=embedding,
                metadata_=metadata or {}
            )
            self.session.add(chunk)
            self.session.flush()
            return chunk
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to create chunk: {str(e)}")

    def bulk_create(self, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """Bulk create multiple chunks.

        This is significantly more efficient than creating chunks one at a time,
        especially for large documents with many chunks.

        Args:
            chunks: List of chunk data dictionaries with keys:
                - document_id: UUID
                - chunk_index: int
                - text: str
                - embedding: List[float] (optional)
                - metadata: dict (optional)

        Returns:
            List of created Chunk entities

        Raises:
            DatabaseConnectionError: If bulk operation fails
        """
        try:
            chunk_objects = [
                Chunk(
                    document_id=chunk["document_id"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    embedding=chunk.get("embedding"),
                    metadata_=chunk.get("metadata", {})
                )
                for chunk in chunks
            ]
            self.session.bulk_save_objects(chunk_objects, return_defaults=True)
            self.session.flush()
            return chunk_objects
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Bulk create failed: {str(e)}")

    def update(self, chunk_id: UUID, **updates) -> Chunk:
        """Update chunk fields.

        Args:
            chunk_id: UUID of the chunk to update
            **updates: Fields to update

        Returns:
            The updated Chunk entity

        Raises:
            EntityNotFoundError: If chunk doesn't exist
            DatabaseConnectionError: If update fails
        """
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)

        try:
            for key, value in updates.items():
                # Handle metadata_ attribute specially
                if key == "metadata":
                    setattr(chunk, "metadata_", value)
                elif hasattr(chunk, key):
                    setattr(chunk, key, value)
            self.session.flush()
            return chunk
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to update chunk: {str(e)}")

    def update_embedding(self, chunk_id: UUID, embedding: List[float]) -> Chunk:
        """Update or add an embedding to a chunk.

        This is a convenience method for the common operation of
        adding embeddings to chunks after text processing.

        Args:
            chunk_id: UUID of the chunk
            embedding: Vector embedding (list of floats)

        Returns:
            The updated Chunk entity

        Raises:
            EntityNotFoundError: If chunk doesn't exist
            DatabaseConnectionError: If update fails
        """
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)

        try:
            chunk.embedding = embedding
            self.session.flush()
            return chunk
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to update embedding: {str(e)}")

    def bulk_update_embeddings(
        self,
        updates: List[Tuple[UUID, List[float]]]
    ) -> int:
        """Bulk update embeddings for multiple chunks.

        This is more efficient than updating embeddings one at a time.

        Args:
            updates: List of (chunk_id, embedding) tuples

        Returns:
            Number of chunks updated

        Raises:
            DatabaseConnectionError: If bulk update fails
        """
        try:
            count = 0
            for chunk_id, embedding in updates:
                result = self.session.query(Chunk).filter(
                    Chunk.chunk_id == chunk_id
                ).update({"embedding": embedding}, synchronize_session=False)
                count += result
            self.session.flush()
            return count
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Bulk update failed: {str(e)}")

    def search_similar(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity.

        Uses cosine distance for similarity calculation. Returns chunks
        ordered by similarity (most similar first).

        Args:
            embedding: Query vector embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (chunk, similarity_score) tuples, where similarity_score
            is 1 - cosine_distance (higher is more similar, 1.0 is identical)

        Raises:
            InvalidQueryError: If embedding is empty or top_k < 1
            DatabaseConnectionError: If search fails
        """
        if not embedding:
            raise InvalidQueryError("Embedding vector cannot be empty")
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")

        try:
            # Convert embedding to pgvector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Query with cosine distance
            # <=> is pgvector's cosine distance operator
            query = text("""
                SELECT chunk_id, document_id, chunk_index, text, embedding,
                       metadata, created_at, updated_at,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> :embedding::vector) >= :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
            """)

            results = self.session.execute(
                query,
                {"embedding": embedding_str, "threshold": threshold, "top_k": top_k}
            ).fetchall()

            # Convert to Chunk objects with similarity scores
            chunks_with_scores = []
            for row in results:
                chunk = Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    embedding=row[4],
                    metadata_=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                # Merge the chunk into the session to track it
                chunk = self.session.merge(chunk)
                similarity = float(row[8])
                chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Vector search failed: {str(e)}")

    def search_similar_with_filter(
        self,
        embedding: List[float],
        top_k: int,
        document_ids: List[UUID],
        threshold: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks filtered by document IDs.

        This is useful when you want to restrict the search to specific
        documents, for example when implementing document-scoped search.

        Args:
            embedding: Query vector embedding
            top_k: Maximum number of results to return
            document_ids: List of document UUIDs to restrict search to
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (chunk, similarity_score) tuples

        Raises:
            InvalidQueryError: If document_ids is empty or embedding is invalid
            DatabaseConnectionError: If search fails
        """
        if not document_ids:
            raise InvalidQueryError("document_ids cannot be empty")
        if not embedding:
            raise InvalidQueryError("Embedding vector cannot be empty")
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")

        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            doc_ids_str = ",".join([f"'{str(doc_id)}'" for doc_id in document_ids])

            query = text(f"""
                SELECT chunk_id, document_id, chunk_index, text, embedding,
                       metadata, created_at, updated_at,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND document_id IN ({doc_ids_str})
                  AND 1 - (embedding <=> :embedding::vector) >= :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
            """)

            results = self.session.execute(
                query,
                {"embedding": embedding_str, "threshold": threshold, "top_k": top_k}
            ).fetchall()

            chunks_with_scores = []
            for row in results:
                chunk = Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    embedding=row[4],
                    metadata_=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                chunk = self.session.merge(chunk)
                similarity = float(row[8])
                chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Filtered vector search failed: {str(e)}")

    def search_similar_with_metadata(
        self,
        embedding: List[float],
        top_k: int,
        metadata_filter: Dict[str, Any],
        threshold: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks filtered by metadata.

        This allows filtering by metadata fields using JSONB operators.

        Args:
            embedding: Query vector embedding
            top_k: Maximum number of results to return
            metadata_filter: Dictionary of metadata field filters
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (chunk, similarity_score) tuples

        Raises:
            InvalidQueryError: If parameters are invalid
            DatabaseConnectionError: If search fails
        """
        if not embedding:
            raise InvalidQueryError("Embedding vector cannot be empty")
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")

        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Build metadata filter conditions
            metadata_conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    metadata_conditions.append(
                        f"metadata->>'{key}' = '{value}'"
                    )
                else:
                    metadata_conditions.append(
                        f"metadata->>'{key}' = '{str(value)}'"
                    )

            where_clause = " AND ".join(metadata_conditions)

            query = text(f"""
                SELECT chunk_id, document_id, chunk_index, text, embedding,
                       metadata, created_at, updated_at,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND {where_clause}
                  AND 1 - (embedding <=> :embedding::vector) >= :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
            """)

            results = self.session.execute(
                query,
                {"embedding": embedding_str, "threshold": threshold, "top_k": top_k}
            ).fetchall()

            chunks_with_scores = []
            for row in results:
                chunk = Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    embedding=row[4],
                    metadata_=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                chunk = self.session.merge(chunk)
                similarity = float(row[8])
                chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Metadata-filtered search failed: {str(e)}")

    def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID.

        Args:
            chunk_id: UUID of the chunk to delete

        Returns:
            True if deletion was successful

        Raises:
            EntityNotFoundError: If chunk doesn't exist
            DatabaseConnectionError: If deletion fails
        """
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)

        try:
            self.session.delete(chunk)
            self.session.flush()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to delete chunk: {str(e)}")

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks belonging to a specific document.

        Args:
            document_id: UUID of the parent document

        Returns:
            Number of chunks deleted

        Raises:
            DatabaseConnectionError: If deletion fails
        """
        try:
            count = self.session.query(Chunk).filter(
                Chunk.document_id == document_id
            ).delete(synchronize_session=False)
            self.session.flush()
            return count
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to delete chunks: {str(e)}")

    # Hierarchy query methods

    def get_parent(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get the parent chunk of a given chunk.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            Parent Chunk if exists, None otherwise

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            chunk = self.get_by_id(chunk_id)
            if not chunk or not chunk.parent_chunk_id:
                return None
            return self.get_by_id(chunk.parent_chunk_id)
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to get parent chunk: {str(e)}")

    def get_children(self, chunk_id: UUID) -> List[Chunk]:
        """Get all child chunks of a given chunk.

        Args:
            chunk_id: UUID of the parent chunk

        Returns:
            List of child Chunk entities ordered by chunk_index

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            return self.session.query(Chunk)\
                .filter(Chunk.parent_chunk_id == chunk_id)\
                .order_by(Chunk.chunk_index)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to get children chunks: {str(e)}")

    def get_siblings(self, chunk_id: UUID) -> List[Chunk]:
        """Get all sibling chunks (chunks with same parent).

        Args:
            chunk_id: UUID of the chunk

        Returns:
            List of sibling Chunk entities (including the chunk itself)

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            chunk = self.get_by_id(chunk_id)
            if not chunk or not chunk.parent_chunk_id:
                return []
            
            return self.session.query(Chunk)\
                .filter(Chunk.parent_chunk_id == chunk.parent_chunk_id)\
                .order_by(Chunk.chunk_index)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to get sibling chunks: {str(e)}")

    def get_ancestors(self, chunk_id: UUID, max_depth: int = 10) -> List[Chunk]:
        """Get all ancestor chunks using recursive query.

        Uses the get_chunk_ancestors database function created in migration.

        Args:
            chunk_id: UUID of the chunk
            max_depth: Maximum depth to traverse (default: 10)

        Returns:
            List of ancestor Chunk entities ordered by depth (closest first)

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            query = text("""
                SELECT chunk_id, parent_chunk_id, hierarchy_level, text, metadata, depth
                FROM get_chunk_ancestors(:chunk_id::uuid, :max_depth)
                WHERE depth > 0
                ORDER BY depth
            """)

            results = self.session.execute(
                query,
                {"chunk_id": str(chunk_id), "max_depth": max_depth}
            ).fetchall()

            ancestors = []
            for row in results:
                chunk = self.get_by_id(UUID(str(row[0])))
                if chunk:
                    ancestors.append(chunk)

            return ancestors

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to get ancestors: {str(e)}")

    def validate_hierarchy(self) -> List[Dict[str, Any]]:
        """Run hierarchy validation queries to detect issues.

        Uses the chunk_hierarchy_validation view created in migration.

        Returns:
            List of validation issues (empty if all OK)

        Raises:
            DatabaseConnectionError: If validation query fails
        """
        try:
            query = text("""
                SELECT 
                    chunk_id,
                    parent_chunk_id,
                    hierarchy_level,
                    document_id,
                    depth,
                    validation_status,
                    validation_message
                FROM chunk_hierarchy_validation
                ORDER BY validation_status DESC, depth
            """)

            results = self.session.execute(query).fetchall()

            issues = []
            for row in results:
                issues.append({
                    "chunk_id": str(row[0]),
                    "parent_chunk_id": str(row[1]) if row[1] else None,
                    "hierarchy_level": row[2],
                    "document_id": str(row[3]),
                    "depth": row[4],
                    "status": row[5],
                    "message": row[6]
                })

            return issues

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Hierarchy validation failed: {str(e)}")
