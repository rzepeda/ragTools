"""Vector index management for PostgreSQL.

This module provides utilities for creating and managing vector indexes
(HNSW, IVFFlat) to optimize similarity search performance.
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import text
from rag_factory.services.interfaces import IDatabaseService

logger = logging.getLogger(__name__)

class VectorIndexManager:
    """Manager for vector indexes in PostgreSQL.
    
    Handles creation and management of HNSW and IVFFlat indexes for
    vector similarity search optimization.
    """
    
    def __init__(self, db_service: IDatabaseService):
        """Initialize vector index manager.
        
        Args:
            db_service: Database service instance
        """
        self.db_service = db_service
        
    async def create_index(
        self,
        index_type: str = 'hnsw',
        metric: str = 'cosine',
        **kwargs
    ) -> None:
        """Create a vector index.
        
        Args:
            index_type: Type of index ('hnsw' or 'ivfflat')
            metric: Distance metric ('cosine', 'l2', 'ip')
            **kwargs: Index-specific parameters
                For HNSW: m (max connections), ef_construction
                For IVFFlat: lists (number of lists)
        """
        table_name = getattr(self.db_service, 'table_name', 'chunks')
        
        # Map metric to operator class
        ops = {
            'cosine': 'vector_cosine_ops',
            'l2': 'vector_l2_ops',
            'ip': 'vector_ip_ops'
        }
        op_class = ops.get(metric, 'vector_cosine_ops')
        
        pool = await self.db_service._get_pool()
        
        async with pool.acquire() as conn:
            if index_type.lower() == 'hnsw':
                m = kwargs.get('m', 16)
                ef = kwargs.get('ef_construction', 64)
                
                query = f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx
                    ON {table_name}
                    USING hnsw (embedding {op_class})
                    WITH (m = {m}, ef_construction = {ef})
                """
                
            elif index_type.lower() == 'ivfflat':
                lists = kwargs.get('lists', 100)
                
                query = f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_ivfflat_idx
                    ON {table_name}
                    USING ivfflat (embedding {op_class})
                    WITH (lists = {lists})
                """
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
                
            logger.info(f"Creating {index_type} index on {table_name}...")
            await conn.execute(query)
            logger.info(f"Successfully created {index_type} index")

    async def drop_index(self, index_type: str = 'hnsw') -> None:
        """Drop a vector index.
        
        Args:
            index_type: Type of index to drop
        """
        table_name = getattr(self.db_service, 'table_name', 'chunks')
        index_name = f"{table_name}_embedding_{index_type}_idx"
        
        pool = await self.db_service._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            logger.info(f"Dropped index {index_name}")

    async def list_indexes(self) -> List[Dict[str, Any]]:
        """List existing indexes on the chunks table.
        
        Returns:
            List of index information dictionaries
        """
        table_name = getattr(self.db_service, 'table_name', 'chunks')
        pool = await self.db_service._get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = '{table_name}'
            """)
            
        return [dict(row) for row in rows]
