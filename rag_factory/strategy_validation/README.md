# Strategy Validation Tool

## Quick Start

```bash
# Test all strategies
python -m rag_factory.strategy_validation.run_validation

# Test specific strategy
python -m rag_factory.strategy_validation.run_validation -s semantic-local-pair

# Custom output file
python -m rag_factory.strategy_validation.run_validation -o results.json
```

## What It Does

Tests each RAG strategy by:
1. Loading strategy pair from `strategies/*.yaml`
2. Indexing test document from `basetext.json` (Voyager 1 example)
3. Running retrieval with test query
4. Outputting results to JSON

## Output Format

```json
{
  "summary": {"total_strategies": 15, "successful": 1, "failed": 14},
  "results": [
    {
      "strategy_name": "semantic-local-pair",
      "indexer": "VectorEmbeddingIndexing",
      "retriever": "SemanticRetriever",
      "query": "What specific item does Voyager 1 carry...",
      "retrieved_chunks": ["text1", "text2"],
      "error": null  // or error message if failed
    }
  ]
}
```

## Current Status (as of 2025-12-21)

- ✅ **1/15 passing**: `semantic-local-pair`
- ❌ **14/15 failing**

## Common Failure Patterns

### 1. Missing Database Tables (11 strategies)
**Error:** `NoSuchTableError: <table_name>`

**Fix:** Run Alembic migrations for the strategy
```bash
alembic upgrade head
```

**Affected:** agentic-rag-pair, context-aware-chunking-pair, contextual-retrieval-pair, fine-tuned-embeddings-pair, keyword-pair, knowledge-graph-pair, multi-query-pair, query-expansion-pair, reranking-pair, self-reflective-pair

### 2. Missing Methods (2 strategies)
**Error:** `AttributeError: 'DatabaseContext' object has no attribute '<method>'`

**Fix:** Implement missing methods in `DatabaseContext`
- `hierarchical-rag-pair`: needs `store_chunks_with_hierarchy()`
- `late-chunking-pair`: needs `index_chunk()`

### 3. Configuration Issues (2 strategies)
**Error:** API authentication or dimension mismatch

**Fix:** 
- `semantic-api-pair`: Set valid `OPENAI_API_KEY` in `.env`
- `hybrid-search-pair`: Fix embedding dimension mismatch (expects 1536, getting 384)

## Files

- `rag_factory/strategy_validation/validate_strategies.py` - Main validation logic
- `rag_factory/strategy_validation/run_validation.py` - CLI entry point
- `rag_factory/strategy_validation/basetext.json` - Test data (7 test cases)

## Configuration

Uses same config as GUI:
- `config/services.yaml` - Service registry
- `.env` - Environment variables
- `strategies/*.yaml` - Strategy definitions
- `alembic.ini` - Database migrations
