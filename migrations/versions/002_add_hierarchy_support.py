"""Add hierarchy support to chunks table

Revision ID: 002
Revises: 001
Create Date: 2025-12-05 11:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add hierarchy support columns to chunks table."""

    # Add parent_chunk_id column (self-referencing foreign key)
    op.add_column(
        "chunks",
        sa.Column("parent_chunk_id", UUID(as_uuid=True), nullable=True,
                  comment="Parent chunk in hierarchy (null for root chunks)")
    )

    # Add hierarchy_level column
    op.add_column(
        "chunks",
        sa.Column("hierarchy_level", sa.Integer(), nullable=False, 
                  server_default="0",
                  comment="Depth in hierarchy: 0=document, 1=section, 2=paragraph, 3=sentence")
    )

    # Add hierarchy_metadata JSONB column
    op.add_column(
        "chunks",
        sa.Column("hierarchy_metadata", JSONB(), nullable=False,
                  server_default="{}",
                  comment="Hierarchy metadata: position_in_parent, total_siblings, depth_from_root")
    )

    # Create foreign key constraint for parent_chunk_id
    op.create_foreign_key(
        "fk_chunks_parent_chunk_id",
        "chunks",
        "chunks",
        ["parent_chunk_id"],
        ["chunk_id"],
        ondelete="CASCADE"
    )

    # Create index on parent_chunk_id for efficient parent/child queries
    op.create_index(
        "idx_chunks_parent_chunk_id",
        "chunks",
        ["parent_chunk_id"]
    )

    # Create composite index for hierarchy queries
    op.create_index(
        "idx_chunks_hierarchy",
        "chunks",
        ["document_id", "hierarchy_level", "chunk_index"]
    )

    # Create view for hierarchy validation
    op.execute("""
        CREATE OR REPLACE VIEW chunk_hierarchy_validation AS
        WITH RECURSIVE hierarchy_check AS (
            -- Base case: root chunks (no parent)
            SELECT 
                chunk_id,
                parent_chunk_id,
                hierarchy_level,
                document_id,
                0 as depth,
                ARRAY[chunk_id] as path,
                'OK' as validation_status,
                NULL as validation_message
            FROM chunks
            WHERE parent_chunk_id IS NULL
            
            UNION ALL
            
            -- Recursive case: child chunks
            SELECT 
                c.chunk_id,
                c.parent_chunk_id,
                c.hierarchy_level,
                c.document_id,
                hc.depth + 1,
                hc.path || c.chunk_id,
                CASE
                    WHEN c.chunk_id = ANY(hc.path) THEN 'ERROR'
                    WHEN c.document_id != hc.document_id THEN 'ERROR'
                    WHEN c.hierarchy_level <= hc.hierarchy_level THEN 'WARNING'
                    WHEN hc.depth >= 10 THEN 'WARNING'
                    ELSE 'OK'
                END,
                CASE
                    WHEN c.chunk_id = ANY(hc.path) THEN 'Circular reference detected'
                    WHEN c.document_id != hc.document_id THEN 'Parent from different document'
                    WHEN c.hierarchy_level <= hc.hierarchy_level THEN 'Child level not greater than parent'
                    WHEN hc.depth >= 10 THEN 'Hierarchy too deep (>10 levels)'
                    ELSE NULL
                END
            FROM chunks c
            INNER JOIN hierarchy_check hc ON c.parent_chunk_id = hc.chunk_id
        )
        SELECT 
            chunk_id,
            parent_chunk_id,
            hierarchy_level,
            document_id,
            depth,
            validation_status,
            validation_message
        FROM hierarchy_check
        WHERE validation_status != 'OK';
    """)

    # Create function to get ancestors
    op.execute("""
        CREATE OR REPLACE FUNCTION get_chunk_ancestors(
            p_chunk_id UUID,
            p_max_depth INTEGER DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id UUID,
            parent_chunk_id UUID,
            hierarchy_level INTEGER,
            text TEXT,
            metadata JSONB,
            depth INTEGER
        )
        LANGUAGE SQL
        STABLE
        AS $$
            WITH RECURSIVE ancestors AS (
                -- Base case: the chunk itself
                SELECT 
                    c.chunk_id,
                    c.parent_chunk_id,
                    c.hierarchy_level,
                    c.text,
                    c.metadata,
                    0 as depth
                FROM chunks c
                WHERE c.chunk_id = p_chunk_id
                
                UNION ALL
                
                -- Recursive case: parent chunks
                SELECT 
                    c.chunk_id,
                    c.parent_chunk_id,
                    c.hierarchy_level,
                    c.text,
                    c.metadata,
                    a.depth + 1
                FROM chunks c
                INNER JOIN ancestors a ON c.chunk_id = a.parent_chunk_id
                WHERE a.depth < p_max_depth
            )
            SELECT * FROM ancestors
            ORDER BY depth;
        $$;
    """)

    # Create function to get descendants
    op.execute("""
        CREATE OR REPLACE FUNCTION get_chunk_descendants(
            p_chunk_id UUID,
            p_max_depth INTEGER DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id UUID,
            parent_chunk_id UUID,
            hierarchy_level INTEGER,
            text TEXT,
            metadata JSONB,
            depth INTEGER
        )
        LANGUAGE SQL
        STABLE
        AS $$
            WITH RECURSIVE descendants AS (
                -- Base case: the chunk itself
                SELECT 
                    c.chunk_id,
                    c.parent_chunk_id,
                    c.hierarchy_level,
                    c.text,
                    c.metadata,
                    0 as depth
                FROM chunks c
                WHERE c.chunk_id = p_chunk_id
                
                UNION ALL
                
                -- Recursive case: child chunks
                SELECT 
                    c.chunk_id,
                    c.parent_chunk_id,
                    c.hierarchy_level,
                    c.text,
                    c.metadata,
                    d.depth + 1
                FROM chunks c
                INNER JOIN descendants d ON c.parent_chunk_id = d.chunk_id
                WHERE d.depth < p_max_depth
            )
            SELECT * FROM descendants
            ORDER BY depth, chunk_index;
        $$;
    """)


def downgrade() -> None:
    """Remove hierarchy support from chunks table."""

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS get_chunk_descendants(UUID, INTEGER)")
    op.execute("DROP FUNCTION IF EXISTS get_chunk_ancestors(UUID, INTEGER)")

    # Drop view
    op.execute("DROP VIEW IF EXISTS chunk_hierarchy_validation")

    # Drop indexes
    op.drop_index("idx_chunks_hierarchy", table_name="chunks")
    op.drop_index("idx_chunks_parent_chunk_id", table_name="chunks")

    # Drop foreign key constraint
    op.drop_constraint("fk_chunks_parent_chunk_id", "chunks", type_="foreignkey")

    # Drop columns
    op.drop_column("chunks", "hierarchy_metadata")
    op.drop_column("chunks", "hierarchy_level")
    op.drop_column("chunks", "parent_chunk_id")
