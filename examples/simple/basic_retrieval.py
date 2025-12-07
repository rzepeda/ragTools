"""
Basic RAG retrieval example using RAG Factory.

This example demonstrates:
- Importing the RAG Factory library
- Creating a simple chunking strategy
- Processing a document
- Displaying results

This is the simplest possible example to get started with RAG Factory.
"""

from rag_factory import RAGFactory
from rag_factory.strategies.chunking import FixedSizeChunker, ChunkingConfig, ChunkingMethod


def main():
    """Run a basic chunking and retrieval example."""
    
    # Sample document
    document = """RAG Factory is a comprehensive library for building RAG applications.
    It provides multiple strategies for chunking, retrieval, and generation.
    You can easily switch between different approaches to find what works best."""
    
    # Create chunking configuration
    config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        target_chunk_size=200,
        chunk_overlap=20
    )
    
    # Initialize chunker
    chunker = FixedSizeChunker(config)
    
    # Process document
    chunks = chunker.chunk_document(document, document_id="sample_doc")
    
    # Display results
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print()


if __name__ == "__main__":
    main()
