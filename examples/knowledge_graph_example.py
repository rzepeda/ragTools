"""
Example usage of Knowledge Graph RAG Strategy.

This example demonstrates how to use the knowledge graph strategy to:
1. Index documents with entity and relationship extraction
2. Perform hybrid retrieval combining vector search and graph traversal
3. View graph statistics and entity relationships
"""

from rag_factory.strategies.knowledge_graph import (
    KnowledgeGraphRAGStrategy,
    KnowledgeGraphConfig
)


def main():
    """Run knowledge graph RAG example."""
    
    # Note: This example requires actual vector store and LLM services
    # For demonstration, we'll show the API usage
    
    print("=" * 60)
    print("Knowledge Graph RAG Strategy Example")
    print("=" * 60)
    
    # Configuration
    config = KnowledgeGraphConfig(
        graph_backend="memory",
        vector_weight=0.6,
        graph_weight=0.4,
        max_graph_hops=2
    )
    
    print("\nConfiguration:")
    print(f"  Graph backend: {config.graph_backend}")
    print(f"  Vector weight: {config.vector_weight}")
    print(f"  Graph weight: {config.graph_weight}")
    print(f"  Max graph hops: {config.max_graph_hops}")
    
    # Initialize strategy (requires actual services)
    # strategy = KnowledgeGraphRAGStrategy(
    #     vector_store_service=vector_store,
    #     llm_service=llm_service,
    #     config=config
    # )
    
    # Example document about Machine Learning
    document = """Python is a popular programming language widely used in Machine Learning.

Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data.

Artificial Intelligence is a broad field focused on creating intelligent machines.

TensorFlow and PyTorch are popular frameworks for Machine Learning development."""
    
    print("\n" + "=" * 60)
    print("Document to Index:")
    print("=" * 60)
    print(document)
    
    # Index document
    print("\n" + "=" * 60)
    print("Indexing Document...")
    print("=" * 60)
    
    # result = strategy.index_document(document, "ml_intro")
    # print(f"\nIndexing Results:")
    # print(f"  Total chunks: {result['total_chunks']}")
    # print(f"  Total entities: {result['total_entities']}")
    # print(f"  Total relationships: {result['total_relationships']}")
    # print(f"\nGraph Statistics:")
    # for key, value in result['graph_stats'].items():
    #     print(f"  {key}: {value}")
    
    # Example retrieval
    print("\n" + "=" * 60)
    print("Hybrid Retrieval Example")
    print("=" * 60)
    
    query = "What frameworks are used for Machine Learning?"
    print(f"\nQuery: {query}")
    
    # results = strategy.retrieve(query, top_k=3)
    # 
    # print(f"\nFound {len(results)} results:\n")
    # for i, result in enumerate(results, 1):
    #     print(f"Result {i}:")
    #     print(f"  Text: {result['text'][:100]}...")
    #     print(f"  Combined Score: {result['score']:.3f}")
    #     print(f"  Vector Score: {result['vector_score']:.3f}")
    #     print(f"  Graph Score: {result['graph_score']:.3f}")
    #     print(f"  Related Entities: {[e['name'] for e in result['related_entities']]}")
    #     if result['relationship_paths']:
    #         print(f"  Relationship Paths: {result['relationship_paths'][:2]}")
    #     print()
    
    print("\nNote: This example shows the API usage.")
    print("To run with actual services, initialize vector_store and llm_service.")
    print("\nExpected behavior:")
    print("  1. Entities extracted: Python, Machine Learning, AI, TensorFlow, PyTorch")
    print("  2. Relationships: Python -> used_for -> ML, ML -> is_part_of -> AI")
    print("  3. Hybrid search combines vector similarity with graph connectivity")
    print("  4. Results include entity information and relationship paths")


if __name__ == "__main__":
    main()
