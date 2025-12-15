"""Create semantic local tables

Revision ID: semantic_local_schema
Revises: 002
Create Date: 2024-12-15
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = 'semantic_local_schema'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade():
    # Chunks table
    op.create_table(
        'semantic_local_chunks',
        sa.Column('chunk_id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Vectors table
    op.create_table(
        'semantic_local_vectors',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('chunk_id', sa.String(255), sa.ForeignKey('semantic_local_chunks.chunk_id')),
        sa.Column('vector_embedding', Vector(384)),  # MiniLM dimension
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Metadata table
    op.create_table(
        'semantic_local_metadata',
        sa.Column('document_id', sa.String(255), primary_key=True),
        sa.Column('title', sa.String(500)),
        sa.Column('source', sa.String(255)),
        sa.Column('metadata', sa.JSON()),
        sa.Column('indexed_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Index for fast vector search
    op.create_index(
        'idx_semantic_local_vectors_embedding',
        'semantic_local_vectors',
        ['vector_embedding'],
        postgresql_using='ivfflat',
        postgresql_ops={'vector_embedding': 'vector_cosine_ops'},
        postgresql_with={'lists': 100}
    )

def downgrade():
    op.drop_table('semantic_local_vectors')
    op.drop_table('semantic_local_chunks')
    op.drop_table('semantic_local_metadata')
