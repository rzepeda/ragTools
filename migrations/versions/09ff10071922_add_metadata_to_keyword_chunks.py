"""add_metadata_to_keyword_chunks

Revision ID: 09ff10071922
Revises: 463c87d677b1
Create Date: 2025-12-21 23:29:19.416561

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '09ff10071922'
down_revision: Union[str, None] = '463c87d677b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add metadata column to keyword_chunks table."""
    op.add_column('keyword_chunks', sa.Column('metadata', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove metadata column from keyword_chunks table."""
    op.drop_column('keyword_chunks', 'metadata')
