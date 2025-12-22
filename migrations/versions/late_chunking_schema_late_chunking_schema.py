"""late_chunking_schema

Revision ID: late_chunking_schema
Revises: agentic_schema
Create Date: 2025-12-19 13:37:45.391668

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'late_chunking_schema'
down_revision: Union[str, None] = 'agentic_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create late chunking strategy tables."""
    # Chunks table
    op.create_table(
        'late_chunking_chunks',
        sa.Column('chunk_id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Vectors table
    op.create_table(
        'late_chunking_vectors',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('chunk_id', sa.String(255), sa.ForeignKey('late_chunking_chunks.chunk_id', ondelete='CASCADE')),
        sa.Column('vector_embedding', Vector(384), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Metadata table
    op.create_table(
        'late_chunking_metadata',
        sa.Column('document_id', sa.String(255), primary_key=True),
        sa.Column('title', sa.String(500)),
        sa.Column('source', sa.String(255)),
        sa.Column('metadata', sa.JSON()),
        sa.Column('indexed_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )


def downgrade() -> None:
    """Drop late chunking strategy tables."""
    op.drop_table('late_chunking_vectors')
    op.drop_table('late_chunking_chunks')
    op.drop_table('late_chunking_metadata')
