# Semantic Local Pair Guide

This guide describes the `semantic-local-pair` strategy configuration, designed for testing and local development without external API dependencies.

## Overview

The Semantic Local Pair implements a standard vector search RAG pipeline:
1.  **Indexing**: Chunks text and generates vector embeddings using a local ONNX model.
2.  **Retrieval**: Embeds queries and searches the vector database using cosine similarity.

## Requirements

- **PostgreSQL** with `pgvector` extension.
- **Python Packages**: `alembic`, `pgvector`, `onnxruntime`, `tokenizers`.
- **System**: 2GB+ RAM recommended for ONNX model execution.

## Configuration

The configuration is defined in `strategies/semantic-local-pair.yaml`.

### Services (`config/services.yaml`)

- `embedding_local`: Uses `Xenova/all-MiniLM-L6-v2` (384 dimensions).
- `db_main`: PostgreSQL database.

### Schema

The strategy uses a dedicated set of tables to avoid conflicts with other strategies:
- `semantic_local_chunks`: Stores text segments.
- `semantic_local_vectors`: Stores embeddings.
- `semantic_local_metadata`: Stores document metadata.

## Setup

1.  **Environment Variables**:
    Ensure `.env` contains:
    ```bash
    DATABASE_URL=postgresql://user:pass@localhost:5432/ragdb
    ```

2.  **Migrations**:
    Apply the database schema:
    ```bash
    alembic upgrade head
    ```

## Usage

### Python API

```python
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.retrieval_interface import RetrievalContext

# 1. Initialize Registry
registry = ServiceRegistry("config/services.yaml")

# 2. Load Strategy Pair
manager = StrategyPairManager(registry, "strategies")
indexing, retrieval = manager.load_pair("semantic-local-pair")

async def run_indexing():
    # 3. Index Documents
    docs = [{"id": "doc1", "text": "RAG is Retrieval Augmented Generation."}]
    context = IndexingContext(indexing.deps.database_service)
    
    # You might need a pre-chunking step if raw docs are passed, 
    # but this strategy expects chunks or handles simple chunking?
    # The VectorEmbeddingIndexer normally expects chunks in DB or input?
    # (Check strategy implementation details)
    await indexing.process(docs, context)


async def run_retrieval():
    # 4. Retrieve
    ret_context = RetrievalContext(retrieval.deps.database_service)
    chunks = await retrieval.retrieve("What is RAG?", ret_context)
    print(chunks)
```

## Troubleshooting

- **Migration Errors**: Ensure `pgvector` is installed in Postgres (`CREATE EXTENSION vector;`).
- **ONNX Errors**: Provide valid `models/embeddings` path or allow download.
