"""
Example usage of the late chunking strategy.

This example demonstrates how to use the late chunking RAG strategy
which embeds full documents before chunking to maintain better context.
"""

import torch
from rag_factory.strategies.late_chunking import (
    LateChunkingRAGStrategy,
    EmbeddingChunkingMethod,
    LateChunkingConfig
)


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""

    def __init__(self):
        self.chunks = []

    def index_chunk(self, chunk_id, text, embedding, metadata):
        """Store a chunk."""
        self.chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        })
        print(f"Indexed chunk: {chunk_id}")

    def search(self, query, top_k=5, **kwargs):
        """Simple search returning first top_k chunks."""
        results = []
        for chunk in self.chunks[:top_k]:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": 0.9,
                "metadata": chunk["metadata"]
            })
        return results


def main():
    """Demonstrate late chunking strategy."""
    print("=" * 80)
    print("Late Chunking Strategy Example")
    print("=" * 80)

    # Create vector store
    vector_store = SimpleVectorStore()

    # Configure late chunking strategy
    config = {
        "model_name": "Xenova/all-MiniLM-L6-v2",
        "chunking_method": EmbeddingChunkingMethod.SEMANTIC_BOUNDARY.value,
        "target_chunk_size": 128,
        "min_chunk_size": 50,
        "max_chunk_size": 256,
        "similarity_threshold": 0.7,
        "compute_coherence_scores": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"\nConfiguration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Chunking method: {config['chunking_method']}")
    print(f"  Target chunk size: {config['target_chunk_size']} tokens")
    print(f"  Device: {config['device']}")

    # Initialize strategy
    print("\nInitializing late chunking strategy...")
    strategy = LateChunkingRAGStrategy(
        vector_store_service=vector_store,
        config=config
    )

    # Sample document
    document = """
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience without being explicitly programmed. 
    It focuses on the development of computer programs that can access data and 
    use it to learn for themselves.
    
    Deep learning is a type of machine learning based on artificial neural networks. 
    The learning process is deep because the structure of artificial neural networks 
    consists of multiple input, output, and hidden layers. Each layer contains units 
    that transform the input data into information that the next layer can use.
    
    Neural networks are computing systems inspired by biological neural networks 
    that constitute animal brains. They are based on a collection of connected units 
    called artificial neurons, which loosely model the neurons in a biological brain.
    
    The training process involves feeding the neural network with training data and 
    adjusting the weights of connections between neurons to minimize the error between 
    predicted and actual outputs. This process is called backpropagation.
    """

    print("\n" + "=" * 80)
    print("Indexing Document")
    print("=" * 80)
    
    # Index the document
    print("\nDocument preview:")
    print(document[:200] + "...")
    
    print("\nIndexing with late chunking...")
    strategy.index_document(document, "ml_doc_001")

    print(f"\nIndexed {len(vector_store.chunks)} chunks")

    # Display chunk information
    print("\n" + "=" * 80)
    print("Chunk Information")
    print("=" * 80)
    
    for i, chunk in enumerate(vector_store.chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Token count: {chunk['metadata']['token_count']}")
        print(f"  Coherence score: {chunk['metadata']['coherence_score']:.3f}")
        print(f"  Token range: {chunk['metadata']['token_range']}")
        print(f"  Text preview: {chunk['text'][:100]}...")

    # Retrieve relevant chunks
    print("\n" + "=" * 80)
    print("Retrieval Example")
    print("=" * 80)
    
    query = "What is deep learning?"
    print(f"\nQuery: {query}")
    
    results = strategy.retrieve(query, top_k=3)
    
    print(f"\nRetrieved {len(results)} chunks:")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Text: {result['text'][:150]}...")

    # Compare chunking methods
    print("\n" + "=" * 80)
    print("Comparing Chunking Methods")
    print("=" * 80)

    methods = [
        ("fixed_size", "Fixed Size"),
        ("semantic_boundary", "Semantic Boundary"),
        ("adaptive", "Adaptive")
    ]

    for method_key, method_name in methods:
        print(f"\n{method_name} Chunking:")
        
        test_store = SimpleVectorStore()
        test_config = config.copy()
        test_config["chunking_method"] = method_key
        test_config["compute_coherence_scores"] = False  # Faster
        
        test_strategy = LateChunkingRAGStrategy(test_store, test_config)
        test_strategy.index_document(document, f"test_{method_key}")
        
        print(f"  Chunks created: {len(test_store.chunks)}")
        if test_store.chunks:
            token_counts = [c["metadata"]["token_count"] for c in test_store.chunks]
            print(f"  Avg tokens/chunk: {sum(token_counts) / len(token_counts):.1f}")
            print(f"  Min tokens: {min(token_counts)}")
            print(f"  Max tokens: {max(token_counts)}")

    print("\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
