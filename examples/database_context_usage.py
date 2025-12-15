"""
DatabaseContext Usage Examples

This file demonstrates how to use DatabaseContext for strategy-specific
table access on a shared database.
"""

from rag_factory.services.database import PostgresqlDatabaseService, DatabaseContext


# ============================================================================
# Example 1: Basic Usage with Single Context
# ============================================================================

def example_basic_usage():
    """Basic usage of DatabaseContext with table and field mapping."""
    
    # Create database service
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="postgres",
        password="password"
    )
    
    # Create context for semantic search strategy
    semantic_context = db_service.get_context(
        table_mapping={
            "chunks": "semantic_chunks",
            "vectors": "semantic_vectors"
        },
        field_mapping={
            "content": "text_content",
            "embedding": "vector_embedding",
            "doc_id": "document_id"
        }
    )
    
    # Use logical names in strategy code
    semantic_context.insert("chunks", {
        "chunk_id": "chunk_001",
        "content": "Python is a programming language",  # Logical name
        "doc_id": "doc123",  # Logical name
        "chunk_index": 0
    })
    # Inserts into semantic_chunks.text_content and semantic_chunks.document_id
    
    # Query using logical names
    results = semantic_context.query(
        "chunks",
        filters={"doc_id": "doc123"},
        limit=10
    )
    
    for row in results:
        print(f"Chunk: {row.text_content}")  # Access via physical name
    
    # Update using logical names
    semantic_context.update(
        "chunks",
        filters={"doc_id": "doc123"},
        updates={"content": "Updated content"}
    )
    
    # Delete using logical names
    semantic_context.delete("chunks", filters={"doc_id": "doc123"})


# ============================================================================
# Example 2: Multiple Strategies Sharing Database
# ============================================================================

def example_multiple_strategies():
    """Multiple strategies using different contexts on same database."""
    
    # Get shared database service
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="postgres",
        password="password"
    )
    
    # Strategy 1: Semantic search
    semantic_context = db_service.get_context(
        table_mapping={
            "chunks": "semantic_chunks",
            "vectors": "semantic_vectors"
        },
        field_mapping={
            "content": "text_content",
            "embedding": "vector_embedding"
        }
    )
    
    # Strategy 2: Keyword search (same DB, different tables)
    keyword_context = db_service.get_context(
        table_mapping={
            "chunks": "keyword_chunks",
            "index": "keyword_inverted_index"
        }
    )
    
    # Both use same connection pool
    assert semantic_context.engine is keyword_context.engine  # True!
    
    # But write to different tables
    semantic_context.insert("chunks", {
        "chunk_id": "sem_1",
        "content": "hello",
        "doc_id": "123"
    })
    # Inserts into semantic_chunks.text_content
    
    keyword_context.insert("chunks", {
        "chunk_id": "key_1",
        "content": "hello",
        "doc_id": "123"
    })
    # Inserts into keyword_chunks.content (different table!)
    
    # Each context only sees its own data
    sem_results = semantic_context.query("chunks")
    key_results = keyword_context.query("chunks")
    
    print(f"Semantic chunks: {len(sem_results)}")  # 1
    print(f"Keyword chunks: {len(key_results)}")   # 1


# ============================================================================
# Example 3: Vector Search with pgvector
# ============================================================================

def example_vector_search():
    """Vector similarity search using DatabaseContext."""
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="postgres",
        password="password"
    )
    
    context = db_service.get_context(
        table_mapping={"vectors": "semantic_vectors"},
        field_mapping={"embedding": "vector_embedding"}
    )
    
    # Insert vectors
    context.insert("vectors", {
        "id": "vec1",
        "embedding": [0.1, 0.2, 0.3] * 128,  # 384-dimensional
        "content": "Python programming"
    })
    
    # Vector search using logical names
    query_vector = [0.15, 0.25, 0.35] * 128
    
    # Cosine distance
    results = context.vector_search(
        "vectors",
        vector_field="embedding",  # Logical name
        query_vector=query_vector,
        top_k=5,
        distance_metric="cosine"
    )
    
    for row, distance in results:
        print(f"Distance: {distance}, Content: {row.content}")
    
    # L2 distance
    results_l2 = context.vector_search(
        "vectors",
        vector_field="embedding",
        query_vector=query_vector,
        top_k=5,
        distance_metric="l2"
    )
    
    # Inner product
    results_ip = context.vector_search(
        "vectors",
        vector_field="embedding",
        query_vector=query_vector,
        top_k=5,
        distance_metric="inner_product"
    )


# ============================================================================
# Example 4: Strategy Implementation with DatabaseContext
# ============================================================================

class SemanticSearchStrategy:
    """Example strategy using DatabaseContext."""
    
    def __init__(self, db_context: DatabaseContext, embedding_service):
        """Initialize with DatabaseContext instead of raw database service.
        
        Args:
            db_context: DatabaseContext with table/field mappings configured
            embedding_service: Service for generating embeddings
        """
        self.db = db_context
        self.embedding_service = embedding_service
    
    def index(self, documents):
        """Index documents using logical table/field names."""
        for doc in documents:
            # Generate embedding
            embedding = self.embedding_service.embed(doc["text"])
            
            # Insert using logical names
            self.db.insert("chunks", {
                "chunk_id": doc["id"],
                "content": doc["text"],  # Logical field name
                "doc_id": doc["document_id"],  # Logical field name
                "embedding": embedding,  # Logical field name
                "chunk_index": doc["index"]
            })
            # DatabaseContext maps to physical names automatically
    
    def search(self, query: str, top_k: int = 5):
        """Search using logical table/field names."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Vector search using logical names
        results = self.db.vector_search(
            "chunks",  # Logical table name
            vector_field="embedding",  # Logical field name
            query_vector=query_embedding,
            top_k=top_k,
            distance_metric="cosine"
        )
        
        return [
            {
                "text": row.text_content,  # Physical field name in result
                "score": 1 - distance,
                "doc_id": row.document_id
            }
            for row, distance in results
        ]


# ============================================================================
# Example 5: Context Caching
# ============================================================================

def example_context_caching():
    """Demonstrate context caching for performance."""
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="postgres",
        password="password"
    )
    
    table_mapping = {"chunks": "test_chunks"}
    field_mapping = {"content": "text"}
    
    # First call creates context
    context1 = db_service.get_context(table_mapping, field_mapping)
    
    # Second call with same mappings returns cached context
    context2 = db_service.get_context(table_mapping, field_mapping)
    
    # Same instance!
    assert context1 is context2
    
    # Different mapping creates new context
    context3 = db_service.get_context({"chunks": "other_chunks"})
    assert context3 is not context1
    
    # But all share same engine (connection pool)
    assert context1.engine is context2.engine is context3.engine


# ============================================================================
# Example 6: Error Handling
# ============================================================================

def example_error_handling():
    """Demonstrate error handling in DatabaseContext."""
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="postgres",
        password="password"
    )
    
    context = db_service.get_context(
        table_mapping={"chunks": "test_chunks", "vectors": "test_vectors"}
    )
    
    # Error: Unmapped table name
    try:
        context.get_table("nonexistent")
    except KeyError as e:
        print(f"Error: {e}")
        # Output: No table mapping for 'nonexistent'. 
        #         Available logical names: ['chunks', 'vectors']
    
    # Error: Invalid distance metric
    try:
        context.vector_search(
            "vectors",
            vector_field="embedding",
            query_vector=[0.1] * 384,
            distance_metric="invalid"
        )
    except ValueError as e:
        print(f"Error: {e}")
        # Output: Unknown distance metric: 'invalid'. 
        #         Valid options: 'cosine', 'l2', 'inner_product'


# ============================================================================
# Example 7: Integration with ServiceRegistry (Epic 17)
# ============================================================================

def example_service_registry_integration():
    """Example of using DatabaseContext with ServiceRegistry."""
    
    # This is how it would work with ServiceRegistry from Story 17.2
    
    # Assume we have a registry with a database service
    # registry = ServiceRegistry()
    # registry.register("db1", db_service)
    
    # Strategy pair configuration (from YAML)
    strategy_config = {
        "strategy_name": "semantic-local-pair",
        "indexer": {
            "strategy": "VectorEmbeddingIndexer",
            "services": {
                "embedding": "$embedding1",
                "db": "$db1"
            },
            "db_config": {
                "tables": {
                    "chunks": "semantic_local_chunks",
                    "vectors": "semantic_local_vectors"
                },
                "fields": {
                    "content": "text_content",
                    "embedding": "vector_embedding",
                    "doc_id": "document_id"
                }
            }
        }
    }
    
    # StrategyPairManager would create context like this:
    # db_service = registry.get("db1")
    # db_context = db_service.get_context(
    #     table_mapping=strategy_config['indexer']['db_config']['tables'],
    #     field_mapping=strategy_config['indexer']['db_config']['fields']
    # )
    # strategy = VectorEmbeddingIndexer(
    #     db=db_context,
    #     embedding_service=registry.get("embedding1")
    # )


if __name__ == "__main__":
    print("DatabaseContext Usage Examples")
    print("=" * 60)
    print("\nSee function docstrings for detailed examples:")
    print("- example_basic_usage()")
    print("- example_multiple_strategies()")
    print("- example_vector_search()")
    print("- example_context_caching()")
    print("- example_error_handling()")
    print("- example_service_registry_integration()")
    print("\nRefer to the implementation for SemanticSearchStrategy")
    print("to see how to use DatabaseContext in a real strategy.")
