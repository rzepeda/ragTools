"""merge all strategy schema heads

Revision ID: 463c87d677b1
Revises: context_aware_schema, contextual_schema, finetuned_schema, hierarchical_schema, keyword_schema, knowledge_graph_schema, late_chunking_schema, multi_query_schema, query_expansion_schema, reranking_schema, self_reflective_schema, semantic_api_schema
Create Date: 2025-12-20 09:19:38.339015

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '463c87d677b1'
down_revision: Union[str, None] = ('context_aware_schema', 'contextual_schema', 'finetuned_schema', 'hierarchical_schema', 'keyword_schema', 'knowledge_graph_schema', 'late_chunking_schema', 'multi_query_schema', 'query_expansion_schema', 'reranking_schema', 'self_reflective_schema', 'semantic_api_schema')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
