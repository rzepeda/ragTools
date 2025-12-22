"""keyword_schema

Revision ID: keyword_schema
Revises: semantic_local_schema
Create Date: 2025-12-19 12:46:59.157182

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'keyword_schema'
down_revision: Union[str, None] = 'semantic_local_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create keyword search strategy tables."""
    # Chunks table
    op.create_table(
        'keyword_chunks',
        sa.Column('chunk_id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Metadata table
    op.create_table(
        'keyword_metadata',
        sa.Column('document_id', sa.String(255), primary_key=True),
        sa.Column('title', sa.String(500)),
        sa.Column('source', sa.String(255)),
        sa.Column('metadata', sa.JSON()),
        sa.Column('indexed_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    
    # Inverted index table for keyword search
    op.create_table(
        'keyword_inverted_index',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('term', sa.String(255), nullable=False, index=True),
        sa.Column('chunk_id', sa.String(255), nullable=False),
        sa.Column('score', sa.Float(), nullable=False, default=1.0),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Create index on term for fast lookup
    op.create_index(
        'idx_keyword_inverted_index_term',
        'keyword_inverted_index',
        ['term']
    )
    
    # Full-text search index
    op.execute("""
        CREATE INDEX idx_keyword_chunks_text_search 
        ON keyword_chunks 
        USING gin(to_tsvector('english', text_content))
    """)


def downgrade() -> None:
    """Drop keyword search strategy tables."""
    op.drop_table('keyword_inverted_index')
    op.drop_table('keyword_chunks')
    op.drop_table('keyword_metadata')
