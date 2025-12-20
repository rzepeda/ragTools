"""agentic_schema

Revision ID: agentic_schema
Revises: semantic_local_schema
Create Date: 2025-12-19 12:46:59.157182

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'agentic_schema'
down_revision: Union[str, None] = 'semantic_local_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
