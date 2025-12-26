"""create agentic_documents and agentic_chunks tables

Revision ID: 84cc75311203
Revises: 09ff10071922
Create Date: 2025-12-25 00:21:03.234084

"""
from typing import Sequence, Union

from pgvector.sqlalchemy import Vector
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84cc75311203'
down_revision: Union[str, None] = '09ff10071922'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'agentic_documents' not in tables:
        op.create_table(
            'agentic_documents',
            sa.Column('document_id', sa.UUID(), nullable=False),
            sa.Column('filename', sa.String(length=255), nullable=False),
            sa.Column('source_path', sa.Text(), nullable=False),
            sa.Column('content_hash', sa.String(length=64), nullable=False),
            sa.Column('total_chunks', sa.Integer(), nullable=False),
            sa.Column('metadata', sa.JSON(), nullable=False),
            sa.Column('status', sa.String(length=50), nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('document_id')
        )
    op.create_index(op.f('ix_agentic_documents_content_hash'), 'agentic_documents', ['content_hash'], unique=False)
    op.create_index(op.f('ix_agentic_documents_filename'), 'agentic_documents', ['filename'], unique=False)
    op.create_index(op.f('ix_agentic_documents_status'), 'agentic_documents', ['status'], unique=False)

    if 'agentic_chunks' not in tables:
        op.create_table(
            'agentic_chunks',
            sa.Column('chunk_id', sa.UUID(), nullable=False),
            sa.Column('document_id', sa.UUID(), nullable=False),
            sa.Column('chunk_index', sa.Integer(), nullable=False),
            sa.Column('text', sa.Text(), nullable=False),
            sa.Column('embedding', Vector(384), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=False),
            sa.Column('parent_chunk_id', sa.UUID(), nullable=True),
            sa.Column('hierarchy_level', sa.Integer(), server_default='0', nullable=False),
            sa.Column('hierarchy_metadata', sa.JSON(), server_default='{}', nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['document_id'], ['agentic_documents.document_id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['parent_chunk_id'], ['agentic_chunks.chunk_id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('chunk_id')
        )
    op.create_index(op.f('ix_agentic_chunks_created_at'), 'agentic_chunks', ['created_at'], unique=False)
    op.create_index(op.f('ix_agentic_chunks_document_id_chunk_index'), 'agentic_chunks', ['document_id', 'chunk_index'], unique=False)
    op.create_index(op.f('ix_agentic_chunks_document_id'), 'agentic_chunks', ['document_id'], unique=False)
    op.create_index(op.f('ix_agentic_chunks_parent_chunk_id'), 'agentic_chunks', ['parent_chunk_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_agentic_chunks_parent_chunk_id'), table_name='agentic_chunks')
    op.drop_index(op.f('ix_agentic_chunks_document_id'), table_name='agentic_chunks')
    op.drop_index(op.f('ix_agentic_chunks_document_id_chunk_index'), table_name='agentic_chunks')
    op.drop_index(op.f('ix_agentic_chunks_created_at'), table_name='agentic_chunks')
    op.drop_table('agentic_chunks')

    op.drop_index(op.f('ix_agentic_documents_status'), table_name='agentic_documents')
    op.drop_index(op.f('ix_agentic_documents_filename'), table_name='agentic_documents')
    op.drop_index(op.f('ix_agentic_documents_content_hash'), table_name='agentic_documents')
    op.drop_table('agentic_documents')
