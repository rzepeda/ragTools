"""contextual_schema

Revision ID: contextual_schema
Revises: semantic_local_schema
Create Date: 2025-12-19 12:46:59.157182

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'contextual_schema'
down_revision: Union[str, None] = 'semantic_local_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create contextual retrieval strategy tables."""
    # Chunks table
    op.create_table(
        'contextual_chunks',
        sa.Column('chunk_id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        if_not_exists=True
    )
    
    # Vectors table
    op.create_table(
        'contextual_vectors',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('chunk_id', sa.String(255), sa.ForeignKey('contextual_chunks.chunk_id')),
        sa.Column('vector_embedding', Vector(384)),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        if_not_exists=True
    )
    
    # Metadata table
    op.create_table(
        'contextual_metadata',
        sa.Column('document_id', sa.String(255), primary_key=True),
        sa.Column('title', sa.String(500)),
        sa.Column('source', sa.String(255)),
        sa.Column('metadata', sa.JSON()),
        sa.Column('indexed_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        if_not_exists=True
    )
    
    # Index for fast vector search
    op.create_index(
        'idx_contextual_vectors_embedding',
        'contextual_vectors',
        ['vector_embedding'],
        postgresql_using='ivfflat',
        postgresql_ops={'vector_embedding': 'vector_cosine_ops'},
        postgresql_with={'lists': 100},
        if_not_exists=True
    )


def downgrade() -> None:
    """Drop contextual retrieval strategy tables."""
    op.drop_table('contextual_vectors')
    op.drop_table('contextual_chunks')
    op.drop_table('contextual_metadata')
