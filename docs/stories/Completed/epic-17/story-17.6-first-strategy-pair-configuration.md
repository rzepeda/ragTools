# Story 17.6: Create First Strategy Pair Configuration (Testing & CLI Integration)

**As a** developer  
**I want** a single, complete strategy pair configuration  
**So that** I can test the entire Epic 17 implementation and integrate with CLI (Epic 14)

## Acceptance Criteria
- Create ONE fully-functional strategy pair: `semantic-local-pair.yaml`
- Uses local ONNX services (no API keys needed for testing)
- Complete `services.yaml` with all required services
- All required Alembic migrations documented and created
- Comprehensive README with step-by-step setup instructions
- Example usage code demonstrating full workflow
- Unit tests validating the configuration loads correctly
- Integration tests showing end-to-end indexing and retrieval
- CLI integration tests (coordinate with Epic 14)
- Performance benchmarks documented

## Why This Strategy First
- ✅ No API keys required (uses local ONNX models)
- ✅ Fast to test (small model, quick loading)
- ✅ Common use case (semantic search is fundamental)
- ✅ Tests all Epic 17 components (ServiceRegistry, DatabaseContext, etc.)
- ✅ Can be used immediately by CLI for smoke tests

## Deliverables

```
config/
└── services.yaml              # Service definitions

strategies/
├── semantic-local-pair.yaml   # The test strategy pair
└── README.md                  # Quick start guide

migrations/versions/
└── xxxx_semantic_local_schema.py  # Alembic migration

tests/integration/
└── test_semantic_local_pair.py    # End-to-end tests

docs/
└── semantic-local-pair-guide.md   # Comprehensive guide
```

## Configuration Files

**config/services.yaml:**
```yaml
# Service Registry - Local ONNX Services (No API Keys Needed)
services:
  embedding_local:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384
  
  db_main:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
```

**strategies/semantic-local-pair.yaml:**
```yaml
strategy_name: "semantic-local-pair"
version: "1.0.0"
description: "Semantic search using local ONNX embeddings (no API keys required)"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"
    db: "$db_main"
  
  db_config:
    tables:
      chunks: "semantic_local_chunks"
      vectors: "semantic_local_vectors"
      metadata: "semantic_local_metadata"
    fields:
      content: "text_content"
      embedding: "vector_embedding"
      doc_id: "document_id"
      chunk_id: "chunk_id"
  
  config:
    chunk_size: 512
    overlap: 50
    min_chunk_size: 100

retriever:
  strategy: "SemanticRetriever"
  services:
    embedding: "$embedding_local"  # Same service as indexer
    db: "$db_main"                # Same database
  
  db_config:
    tables:
      vectors: "semantic_local_vectors"
    fields:
      embedding: "vector_embedding"
      content: "text_content"
  
  config:
    top_k: 5
    similarity_threshold: 0.7
    distance_metric: "cosine"

migrations:
  required_revisions:
    - "semantic_local_schema"  # Alembic revision ID

expected_schema:
  tables:
    - "semantic_local_chunks"
    - "semantic_local_vectors"
    - "semantic_local_metadata"
  indexes:
    - "idx_semantic_local_vectors_embedding"
  extensions:
    - "vector"  # pgvector

tags:
  - "semantic"
  - "local"
  - "onnx"
  - "testing"
```

## Alembic Migration
```python
# migrations/versions/xxxx_semantic_local_schema.py
"""Create semantic local tables

Revision ID: semantic_local_schema
Revises: base_schema
Create Date: 2024-12-15
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = 'semantic_local_schema'
down_revision = 'base_schema'

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
```

## Story Points
5
