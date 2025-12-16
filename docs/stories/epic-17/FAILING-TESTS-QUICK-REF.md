# Story 17.7 - Failing Tests Quick Reference

## Current Status: 5/15 Passing (33%)

### ✅ Passing Tests
- test_semantic_local_pair.py
- test_semantic_api_pair.py
- test_fine_tuned_embeddings_pair.py
- test_self_reflective_pair.py
- test_agentic_rag_pair.py

### ❌ Failing Tests (10)

#### 1. test_keyword_pair.py
**Error**: Tests hang during execution
**Steps Taken**:
- Created KeywordRetriever strategy (172 lines)
- Registered KeywordIndexer with factory
- Fixed YAML service keys (database → db)
**Recommendation**: Check for missing async methods in mock database service (search_keyword, get_all_chunks)

#### 2. test_hierarchical_rag_pair.py
**Error**: `assert False - isinstance(HierarchicalRAGStrategy, IIndexingStrategy)`
**Steps Taken**:
- Fixed constructor to accept config/dependencies
- Registered with factory
**Recommendation**: Update test to handle IRAGStrategy (not IIndexingStrategy) - this is a full RAG strategy

#### 3. test_hybrid_search_pair.py
**Error**: `Strategy 'HybridIndexer' not found`
**Steps Taken**:
- Fixed YAML service keys
**Recommendation**: Create HybridIndexer or update YAML to use VectorEmbeddingIndexer

#### 4. test_knowledge_graph_pair.py
**Error**: `KnowledgeGraphRAGStrategy requires services: GRAPH`
**Steps Taken**:
- Registered strategy with factory
- Fixed constructor
**Recommendation**: Add mock graph service to test fixture

#### 5. test_late_chunking_pair.py
**Error**: `Retrieval requires VECTORS but indexing produces set()`
**Steps Taken**:
- Fixed constructor to accept config/dependencies
**Recommendation**: Add produces() method to LateChunkingStrategy returning {VECTORS, DATABASE}

#### 6. test_multi_query_pair.py
**Error**: `MultiQueryRAGStrategy requires services: DATABASE`
**Steps Taken**:
- Registered strategy with factory
- Fixed YAML service keys (database → db)
**Recommendation**: Verify test mock provides db_main service correctly

#### 7. test_query_expansion_pair.py
**Error**: `SemanticRetriever: Missing required services: DATABASE`
**Steps Taken**:
- Updated YAML to use SemanticRetriever
- Fixed service keys (database → db)
**Recommendation**: Verify test mock provides all required services

#### 8. test_reranking_pair.py
**Error**: `SemanticRetriever: Missing required services: DATABASE`
**Steps Taken**:
- Updated YAML to use SemanticRetriever
- Fixed service keys (database → db)
**Recommendation**: Same as query_expansion - verify mock services

#### 9. test_contextual_retrieval_pair.py
**Error**: `Retrieval requires VECTORS but indexing produces set()`
**Steps Taken**:
- Registered ContextualRetrievalStrategy
**Recommendation**: Check if ContextualRetrievalStrategy implements produces() correctly

#### 10. test_context_aware_chunking_pair.py
**Error**: `Retrieval requires VECTORS but indexing produces CHUNKS, DATABASE`
**Steps Taken**:
- Registered ContextAwareChunker
- Added produces() method
**Recommendation**: Either add VECTORS to ContextAwareChunker.produces() or pair with VectorEmbeddingIndexer

## Quick Fix Priority

### High Priority (Quick Wins)
1. **Add produces() methods** (tests 5, 9, 10)
   - LateChunkingStrategy
   - ContextualRetrievalStrategy
   - Fix ContextAwareChunker capabilities

2. **Fix test type assertions** (test 2)
   - Update test_hierarchical_rag_pair.py to expect IRAGStrategy

3. **Add mock services** (test 4)
   - Add graph_service to test fixture

### Medium Priority
4. **Create HybridIndexer** (test 3)
   - Or update YAML to use existing strategy

5. **Debug async hangs** (tests 1, 6, 7, 8)
   - Add missing async methods to mocks
   - Check for infinite loops in strategies

## Common Patterns

**Service Key Issue**: Many YAMLs used `database:` but should use `db:`
**Solution**: Changed all to `db: "$db_main"`

**Constructor Issue**: Many strategies had old-style constructors
**Solution**: Updated to accept `config` and `dependencies` parameters

**Registration Issue**: Strategies not discoverable by factory
**Solution**: Added `@register_rag_strategy("StrategyName")` decorators

## Next Steps

1. Add produces() methods (15 min)
2. Fix test type assertions (5 min)
3. Add mock graph service (10 min)
4. Create or update HybridIndexer (20 min)
5. Debug async hangs (30-60 min)

**Total estimated time to 100%**: 1.5-2 hours
