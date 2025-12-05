"""Repository for Document entity operations.

This module implements the repository pattern for Document entities,
providing CRUD operations, pagination, and status filtering.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .base import BaseRepository
from .exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    DatabaseConnectionError
)
from ..database.models import Document


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document CRUD operations.

    This repository provides methods for creating, reading, updating,
    and deleting documents, as well as specialized queries for
    deduplication and status filtering.
    """

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Retrieve a document by its ID.

        Args:
            document_id: UUID of the document to retrieve

        Returns:
            Document if found, None otherwise

        Raises:
            DatabaseConnectionError: If database query fails
        """
        try:
            return self.session.query(Document).filter(
                Document.document_id == document_id
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query document: {str(e)}")

    def get_by_content_hash(self, content_hash: str) -> Optional[Document]:
        """Retrieve a document by its content hash.

        This method is useful for deduplication, allowing you to check
        if a document with the same content already exists.

        Args:
            content_hash: SHA-256 hash of the document content

        Returns:
            Document if found, None otherwise

        Raises:
            DatabaseConnectionError: If database query fails
        """
        try:
            return self.session.query(Document).filter(
                Document.content_hash == content_hash
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query document: {str(e)}")

    def create(
        self,
        filename: str,
        source_path: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "pending"
    ) -> Document:
        """Create a new document.

        This method checks for duplicate content hashes before creating
        the document to prevent duplicates.

        Args:
            filename: Name of the document file
            source_path: Path or URL where the document originated
            content_hash: SHA-256 hash of the document content
            metadata: Optional dictionary of custom metadata
            status: Processing status (default: "pending")

        Returns:
            The created Document entity

        Raises:
            DuplicateEntityError: If document with same content_hash exists
            DatabaseConnectionError: If database operation fails
        """
        # Check for duplicates
        existing = self.get_by_content_hash(content_hash)
        if existing:
            raise DuplicateEntityError("Document", "content_hash", content_hash)

        try:
            document = Document(
                filename=filename,
                source_path=source_path,
                content_hash=content_hash,
                metadata_=metadata or {},
                status=status
            )
            self.session.add(document)
            self.session.flush()
            return document
        except IntegrityError as e:
            self.session.rollback()
            raise DuplicateEntityError("Document", "unknown", str(e))
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to create document: {str(e)}")

    def bulk_create(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Bulk create multiple documents.

        This is more efficient than creating documents one at a time
        for large batches.

        Args:
            documents: List of document data dictionaries with keys:
                - filename: str
                - source_path: str
                - content_hash: str
                - metadata: dict (optional)
                - status: str (optional, default: "pending")

        Returns:
            List of created Document entities

        Raises:
            DatabaseConnectionError: If bulk operation fails
        """
        try:
            doc_objects = [
                Document(
                    filename=doc["filename"],
                    source_path=doc["source_path"],
                    content_hash=doc["content_hash"],
                    metadata_=doc.get("metadata", {}),
                    status=doc.get("status", "pending")
                )
                for doc in documents
            ]
            self.session.bulk_save_objects(doc_objects, return_defaults=True)
            self.session.flush()
            return doc_objects
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Bulk create failed: {str(e)}")

    def update(self, document_id: UUID, **updates) -> Document:
        """Update document fields.

        Args:
            document_id: UUID of the document to update
            **updates: Fields to update (e.g., filename="new.txt", status="completed")

        Returns:
            The updated Document entity

        Raises:
            EntityNotFoundError: If document doesn't exist
            DatabaseConnectionError: If update operation fails
        """
        document = self.get_by_id(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        try:
            for key, value in updates.items():
                # Handle metadata_ attribute specially
                if key == "metadata":
                    setattr(document, "metadata_", value)
                elif hasattr(document, key):
                    setattr(document, key, value)
            self.session.flush()
            return document
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to update document: {str(e)}")

    def update_status(self, document_id: UUID, status: str) -> Document:
        """Update only the document status.

        This is a convenience method for the common operation of
        updating a document's processing status.

        Args:
            document_id: UUID of the document
            status: New status value

        Returns:
            The updated Document entity

        Raises:
            EntityNotFoundError: If document doesn't exist
            DatabaseConnectionError: If update fails
        """
        return self.update(document_id, status=status)

    def delete(self, document_id: UUID) -> bool:
        """Delete a document.

        Due to cascade settings, this will also delete all associated chunks.

        Args:
            document_id: UUID of the document to delete

        Returns:
            True if deletion was successful

        Raises:
            EntityNotFoundError: If document doesn't exist
            DatabaseConnectionError: If deletion fails
        """
        document = self.get_by_id(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        try:
            self.session.delete(document)
            self.session.flush()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Failed to delete document: {str(e)}")

    def bulk_delete(self, document_ids: List[UUID]) -> int:
        """Delete multiple documents by their IDs.

        Args:
            document_ids: List of document UUIDs to delete

        Returns:
            Number of documents deleted

        Raises:
            DatabaseConnectionError: If deletion fails
        """
        try:
            count = self.session.query(Document).filter(
                Document.document_id.in_(document_ids)
            ).delete(synchronize_session=False)
            self.session.flush()
            return count
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseConnectionError(f"Bulk delete failed: {str(e)}")

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List documents with pagination.

        Documents are ordered by creation date (newest first).

        Args:
            skip: Number of documents to skip (for pagination)
            limit: Maximum number of documents to return

        Returns:
            List of Document entities

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            return self.session.query(Document)\
                .order_by(Document.created_at.desc())\
                .offset(skip)\
                .limit(limit)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to list documents: {str(e)}")

    def count(self) -> int:
        """Count total number of documents.

        Returns:
            Total number of documents in the database

        Raises:
            DatabaseConnectionError: If count query fails
        """
        try:
            return self.session.query(func.count(Document.document_id)).scalar()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to count documents: {str(e)}")

    def get_by_status(self, status: str) -> List[Document]:
        """Retrieve all documents with a specific status.

        Args:
            status: Status to filter by (e.g., "pending", "completed", "failed")

        Returns:
            List of documents with the specified status

        Raises:
            DatabaseConnectionError: If query fails
        """
        try:
            return self.session.query(Document).filter(
                Document.status == status
            ).all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query by status: {str(e)}")
