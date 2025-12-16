"""Database module for RAG Factory.

This module provides database connectivity, ORM models, and utilities
for working with PostgreSQL and pgvector.
"""

from rag_factory.database.config import DatabaseConfig
from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.models import Base, Document, Chunk

__all__ = [
    "DatabaseConfig",
    "DatabaseConnection",
    "Base",
    "Document",
    "Chunk",
]
