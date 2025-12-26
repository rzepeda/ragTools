"""add_embedding_column_to_agentic_chunks

Revision ID: d4d5eec9c27f
Revises: 84cc75311203
Create Date: 2025-12-26 02:04:12.716339

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'd4d5eec9c27f'
down_revision: Union[str, None] = '84cc75311203'




def upgrade() -> None:
    op.execute("ALTER TABLE agentic_chunks ADD COLUMN IF NOT EXISTS embedding VECTOR(384)")


def downgrade() -> None:
    op.execute("ALTER TABLE agentic_chunks DROP COLUMN IF EXISTS embedding")
