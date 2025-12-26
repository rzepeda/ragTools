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
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='parent_chunk_id') THEN
                ALTER TABLE chunks ADD COLUMN parent_chunk_id UUID NULL;
            END IF;
        END
        $$;
    """)

    # Add hierarchy_level column
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='hierarchy_level') THEN
                ALTER TABLE chunks ADD COLUMN hierarchy_level INTEGER NOT NULL DEFAULT 0;
            END IF;
        END
        $$;
    """)

    # Add hierarchy_metadata JSONB column
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='hierarchy_metadata') THEN
                ALTER TABLE chunks ADD COLUMN hierarchy_metadata JSONB NOT NULL DEFAULT '{}';
            END IF;
        END
        $$;
    """)

    # Create foreign key constraint for parent_chunk_id
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_chunks_parent_chunk_id') THEN
                ALTER TABLE chunks ADD CONSTRAINT fk_chunks_parent_chunk_id FOREIGN KEY (parent_chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE;
            END IF;
        END
        $$;
    """)

    # Create index on parent_chunk_id for efficient parent/child queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id
        ON chunks (parent_chunk_id);
    """)

    # Create composite index for hierarchy queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy
        ON chunks (document_id, hierarchy_level, chunk_index);
    """)

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
            chunk_index INTEGER,
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
                    c.chunk_index,
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
                    c.chunk_index,
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

    # Drop indexes (with IF EXISTS to handle cases where they don't exist)
    op.execute("DROP INDEX IF EXISTS idx_chunks_hierarchy")
    op.execute("DROP INDEX IF EXISTS idx_chunks_parent_chunk_id")

    # Drop foreign key constraint (with table existence check)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks') THEN
                ALTER TABLE chunks DROP CONSTRAINT IF EXISTS fk_chunks_parent_chunk_id;
            END IF;
        END $$;
    """)

    # Drop columns (with table existence check)
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks') THEN
                ALTER TABLE chunks DROP COLUMN IF EXISTS hierarchy_metadata;
                ALTER TABLE chunks DROP COLUMN IF EXISTS hierarchy_level;
                ALTER TABLE chunks DROP COLUMN IF EXISTS parent_chunk_id;
            END IF;
        END $$;
    """)
