"""SQLAlchemy ORM models for RAG Factory database.

This module defines the database schema for documents and chunks,
including support for vector embeddings using pgvector.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Integer, Text, TIMESTAMP, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, mapped_column
from sqlalchemy.types import TypeDecorator, CHAR
from pgvector.sqlalchemy import Vector
import json


Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            else:
                return value


class JSONType(TypeDecorator):
    """Platform-independent JSON type.

    Uses PostgreSQL's JSONB type, otherwise uses TEXT, storing as JSON strings.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        if dialect.name == 'postgresql':
            return value
        else:
            return json.loads(value) if value else {}


class Document(Base):
    """Document metadata model.

    Stores metadata about documents that have been processed and chunked.
    Each document can have multiple associated chunks.

    Attributes:
        document_id: Unique identifier for the document
        filename: Original filename
        source_path: Path or URL where the document was sourced
        content_hash: SHA-256 hash for deduplication
        total_chunks: Number of chunks created from this document
        metadata: Flexible JSONB field for custom metadata
        status: Processing status (pending, processing, completed, failed)
        created_at: Timestamp when document was created
        updated_at: Timestamp when document was last updated
        chunks: Relationship to associated Chunk records
    """

    __tablename__ = "documents"

    document_id = Column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )

    filename = Column(
        String(255),
        nullable=False,
        index=True
    )

    source_path = Column(
        Text,
        nullable=False
    )

    content_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash for deduplication"
    )

    total_chunks = Column(
        Integer,
        default=0,
        nullable=False
    )

    metadata_ = Column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
        comment="Flexible metadata storage"
    )

    status = Column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
        comment="Processing status: pending, processing, completed, failed"
    )

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        default=datetime.utcnow,
        server_default="NOW()"
    )

    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default="NOW()"
    )

    # Relationship to chunks
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    def __repr__(self) -> str:
        """String representation of Document."""
        return (
            f"<Document(id={self.document_id}, "
            f"filename='{self.filename}', "
            f"status='{self.status}', "
            f"chunks={self.total_chunks})>"
        )


class Chunk(Base):
    """Text chunk model with vector embeddings.

    Stores individual text chunks from documents along with their
    vector embeddings for similarity search.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Foreign key to parent document
        chunk_index: Order of chunk within the document (0-indexed)
        text: The actual text content of the chunk
        embedding: Vector embedding for similarity search (nullable)
        metadata: Flexible JSONB field for custom metadata
        parent_chunk_id: Foreign key to parent chunk in hierarchy (nullable)
        hierarchy_level: Depth in hierarchy (0=document, 1=section, 2=paragraph, 3=sentence)
        hierarchy_metadata: JSONB with position_in_parent, total_siblings, depth_from_root
        created_at: Timestamp when chunk was created
        updated_at: Timestamp when chunk was last updated
        document: Relationship to parent Document record
        parent: Relationship to parent Chunk record (nullable)
        children: Relationship to child Chunk records
    """

    __tablename__ = "chunks"

    chunk_id = Column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )

    document_id = Column(
        GUID,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    chunk_index = Column(
        Integer,
        nullable=False,
        comment="Order within document (0-indexed)"
    )

    text = Column(
        Text,
        nullable=False
    )

    embedding = Column(
        Vector(1536),
        nullable=True,
        comment="Vector embedding for similarity search"
    )

    metadata_ = Column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
        comment="Flexible metadata storage"
    )

    # Hierarchy support columns
    parent_chunk_id = Column(
        GUID,
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Parent chunk in hierarchy (null for root chunks)"
    )

    hierarchy_level = Column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Depth in hierarchy: 0=document, 1=section, 2=paragraph, 3=sentence"
    )

    hierarchy_metadata = Column(
        "hierarchy_metadata",
        JSONType,
        default=dict,
        nullable=False,
        server_default="{}",
        comment="Hierarchy metadata: position_in_parent, total_siblings, depth_from_root"
    )

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        default=datetime.utcnow,
        server_default="NOW()"
    )

    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default="NOW()"
    )

    # Relationship to document
    document = relationship(
        "Document",
        back_populates="chunks"
    )

    # Hierarchy relationships
    parent = relationship(
        "Chunk",
        remote_side=[chunk_id],
        back_populates="children",
        foreign_keys=[parent_chunk_id]
    )

    children = relationship(
        "Chunk",
        back_populates="parent",
        foreign_keys=[parent_chunk_id],
        cascade="all, delete-orphan"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_chunks_document_id_index", "document_id", "chunk_index"),
        Index("idx_chunks_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of Chunk."""
        has_embedding = "yes" if self.embedding is not None else "no"
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"<Chunk(id={self.chunk_id}, "
            f"doc_id={self.document_id}, "
            f"index={self.chunk_index}, "
            f"embedding={has_embedding}, "
            f"text='{text_preview}')>"
        )
