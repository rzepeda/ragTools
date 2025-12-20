"""context_aware_schema

Revision ID: context_aware_schema
Revises: agentic_schema
Create Date: 2025-12-19 13:34:26.238149

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'context_aware_schema'
down_revision: Union[str, None] = 'agentic_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
