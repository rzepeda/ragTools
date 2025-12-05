"""Example demonstrating different chunking strategies.

This script shows how to use the various chunking strategies
available in the rag_factory library.
"""

from rag_factory.strategies.chunking import (
    StructuralChunker,
    FixedSizeChunker,
    HybridChunker,
    ChunkingConfig,
    ChunkingMethod
)


def print_chunk_info(chunks, strategy_name):
    """Print information about chunks."""
    print(f"\n{'='*60}")
    print(f"{strategy_name}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk.metadata.chunk_id}")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Hierarchy: {chunk.metadata.section_hierarchy}")
        print(f"  Preview: {chunk.text[:100]}...")


def main():
    """Run chunking examples."""

    # Sample document
    document = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval with generation.

## Key Components

### Vector Database

The vector database stores embeddings of document chunks.
It enables semantic search over large document collections.

### Embedding Model

Embeddings convert text into dense vectors.
These vectors capture semantic meaning.

## Benefits

RAG provides several advantages:
- Reduces hallucinations
- Enables knowledge grounding
- Allows dynamic knowledge updates

# Conclusion

RAG is a powerful technique for building knowledge-grounded applications.
"""

    # Example 1: Structural Chunking (recommended for markdown)
    print("\n" + "="*60)
    print("EXAMPLE 1: Structural Chunking")
    print("="*60)

    structural_config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=256,
        respect_headers=True,
        respect_paragraphs=True
    )

    structural_chunker = StructuralChunker(structural_config)
    structural_chunks = structural_chunker.chunk_document(document, "example_doc")

    print_chunk_info(structural_chunks, "Structural Chunking Results")

    # Get statistics
    stats = structural_chunker.get_stats(structural_chunks)
    print(f"\nStatistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Average size: {stats['avg_chunk_size']:.1f} tokens")
    print(f"  Min size: {stats['min_chunk_size']} tokens")
    print(f"  Max size: {stats['max_chunk_size']} tokens")

    # Example 2: Fixed-Size Chunking (fast baseline)
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Fixed-Size Chunking")
    print("="*60)

    fixed_config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        min_chunk_size=50,
        max_chunk_size=300,
        target_chunk_size=150,
        chunk_overlap=20
    )

    fixed_chunker = FixedSizeChunker(fixed_config)
    fixed_chunks = fixed_chunker.chunk_document(document, "example_doc")

    print_chunk_info(fixed_chunks, "Fixed-Size Chunking Results")

    # Example 3: Hybrid Chunking (without embeddings)
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Hybrid Chunking (Structural Only)")
    print("="*60)
    print("Note: This uses structural chunking only since no embedding service is provided")

    hybrid_config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        target_chunk_size=256,
        use_embeddings=False  # Disable semantic refinement
    )

    hybrid_chunker = HybridChunker(hybrid_config, embedding_service=None)
    hybrid_chunks = hybrid_chunker.chunk_document(document, "example_doc")

    print_chunk_info(hybrid_chunks, "Hybrid Chunking Results")

    # Example 4: Batch Processing
    print("\n\n" + "="*60)
    print("EXAMPLE 4: Batch Document Processing")
    print("="*60)

    documents = [
        {"text": "# Document 1\n\nFirst document content.", "id": "doc_1"},
        {"text": "# Document 2\n\nSecond document content.", "id": "doc_2"},
        {"text": "# Document 3\n\nThird document content.", "id": "doc_3"}
    ]

    batch_results = structural_chunker.chunk_documents(documents)

    print(f"\nProcessed {len(batch_results)} documents:")
    for i, chunks in enumerate(batch_results):
        print(f"  Document {i+1}: {len(chunks)} chunks")

    # Example 5: Comparing Strategies
    print("\n\n" + "="*60)
    print("EXAMPLE 5: Comparing Strategies")
    print("="*60)

    print("\nStrategy Comparison:")
    print(f"  Structural: {len(structural_chunks)} chunks")
    print(f"  Fixed-Size: {len(fixed_chunks)} chunks")
    print(f"  Hybrid:     {len(hybrid_chunks)} chunks")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
