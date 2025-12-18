"""Initial schema with documents and chunks tables

Revision ID: 001
Revises:
Create Date: 2024-12-03 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create documents and chunks tables with pgvector support."""

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("document_id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("filename", sa.String(255), nullable=False, index=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True,
                  comment="SHA-256 hash for deduplication"),
        sa.Column("total_chunks", sa.Integer(), nullable=False, default=0),
        sa.Column("metadata", JSONB(), nullable=False, default={},
                  comment="Flexible metadata storage"),
        sa.Column("status", sa.String(50), nullable=False, default="pending", index=True,
                  comment="Processing status: pending, processing, completed, failed"),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False,
                  server_default=sa.text("NOW()"))
    )

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("chunk_id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False,
                  comment="Order within document (0-indexed)"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True,
                  comment="Vector embedding for similarity search"),
        sa.Column("metadata", JSONB(), nullable=False, default={},
                  comment="Flexible metadata storage"),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            ondelete="CASCADE"
        )
    )

    # Create additional indexes for performance
    op.create_index("idx_chunks_document_id_index", "chunks",
                    ["document_id", "chunk_index"])
    op.create_index("idx_chunks_created_at", "chunks", ["created_at"])

    # Create HNSW index for vector similarity search
    # Using cosine distance for semantic similarity
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Create trigger to automatically update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    op.execute("""
        CREATE TRIGGER update_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_chunks_updated_at
        BEFORE UPDATE ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """Drop all tables and extensions."""

    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_chunks_updated_at ON chunks")
    op.execute("DROP TRIGGER IF EXISTS update_documents_updated_at ON documents")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop indexes (with IF EXISTS to handle cases where they don't exist)
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS idx_chunks_created_at")
    op.execute("DROP INDEX IF EXISTS idx_chunks_document_id_index")

    # Drop tables (with IF EXISTS to handle cases where they don't exist)
    op.execute("DROP TABLE IF EXISTS chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")

    # Drop extension (CASCADE to drop dependent objects like vector columns)
    op.execute("DROP EXTENSION IF EXISTS vector CASCADE")
