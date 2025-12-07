"""
Hierarchical RAG Strategy Example.

This example demonstrates how to use the hierarchical RAG strategy
to index documents with parent-child relationships and retrieve
chunks with expanded context.
"""

from rag_factory.strategies.hierarchical import (
    HierarchicalRAGStrategy,
    ExpansionStrategy,
    HierarchicalConfig
)
from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.config import DatabaseConfig
from rag_factory.services.embedding import EmbeddingService


def main():
    """Run hierarchical RAG example."""
    
    # Setup database connection
    db_config = DatabaseConfig(
        database_url="postgresql://user:pass@localhost/rag_db",
        pool_size=5
    )
    db_connection = DatabaseConnection(db_config)
    
    # Setup embedding service (mock for example)
    class MockEmbeddingService:
        def embed_text(self, text: str):
            # Return mock embedding
            return [0.1] * 1536
    
    embedding_service = MockEmbeddingService()
    
    # Create hierarchical strategy with configuration
    strategy = HierarchicalRAGStrategy(
        vector_store_service=embedding_service,
        database_service=db_connection,
        config={
            "expansion_strategy": ExpansionStrategy.IMMEDIATE_PARENT,
            "small_chunk_size": 256,
            "large_chunk_size": 1024,
            "search_small_chunks": True
        }
    )
    
    # Example markdown document
    markdown_doc = """# Machine Learning Guide

## Introduction

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

### Definition

Machine learning focuses on the development of computer programs that can
access data and use it to learn for themselves.

### History

The term "machine learning" was coined by Arthur Samuel in 1959. Since then,
the field has evolved significantly with advances in computing power and data
availability.

## Types of Learning

### Supervised Learning

Supervised learning uses labeled data to train models. The algorithm learns
from a training dataset with known outcomes and makes predictions on new data.

#### Examples

- Classification: Spam detection, image recognition
- Regression: Price prediction, weather forecasting

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. The algorithm explores
the data structure without predefined labels.

#### Examples

- Clustering: Customer segmentation, anomaly detection
- Dimensionality reduction: Feature extraction, visualization

## Applications

Machine learning has numerous real-world applications across industries:

- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles, route optimization
- Entertainment: Recommendation systems, content generation
"""

    # Index the document with hierarchy
    print("Indexing document with hierarchical structure...")
    strategy.index_document(markdown_doc, "ml_guide_001")
    print("✓ Document indexed successfully\n")
    
    # Test different expansion strategies
    strategies = [
        ExpansionStrategy.IMMEDIATE_PARENT,
        ExpansionStrategy.FULL_SECTION,
        ExpansionStrategy.ADAPTIVE
    ]
    
    query = "What is supervised learning?"
    
    print(f"Query: '{query}'\n")
    print("=" * 80)
    
    for exp_strategy in strategies:
        print(f"\nExpansion Strategy: {exp_strategy.value}")
        print("-" * 80)
        
        # Update strategy config
        strategy.hierarchical_config.expansion_strategy = exp_strategy
        
        # Retrieve with expansion
        results = strategy.retrieve(query, top_k=3)
        
        print(f"Retrieved {len(results)} chunks:\n")
        
        for i, chunk in enumerate(results, 1):
            print(f"Chunk {i}:")
            print(f"  Score: {chunk.score:.3f}")
            print(f"  Original text: {chunk.metadata.get('original_text', '')[:100]}...")
            print(f"  Expanded text length: {len(chunk.text)} chars")
            print(f"  Total tokens: {chunk.metadata.get('total_tokens', 0)}")
            print(f"  Parent chunks: {len(chunk.metadata.get('parent_chunks', []))}")
            print()
    
    # Demonstrate hierarchy validation
    print("\n" + "=" * 80)
    print("Hierarchy Validation")
    print("-" * 80)
    
    issues = db_connection.chunk_repository.validate_hierarchy()
    if issues:
        print(f"Found {len(issues)} validation issues:")
        for issue in issues:
            print(f"  - {issue['status']}: {issue['message']}")
    else:
        print("✓ No hierarchy validation issues found")
    
    # Cleanup
    db_connection.close()
    print("\n✓ Example completed successfully")


if __name__ == "__main__":
    main()
