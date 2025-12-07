"""
Medium complexity example using multiple chunking strategies.

This example demonstrates:
- Using multiple chunking strategies
- Comparing different approaches
- Configuration via YAML file
- Performance timing
- Comprehensive error handling
"""

import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

from rag_factory.strategies.chunking import (
    StructuralChunker,
    FixedSizeChunker,
    HybridChunker,
    ChunkingConfig,
    ChunkingMethod,
    Chunk
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_chunkers(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create multiple chunking strategies for comparison.
    
    Returns:
        Dictionary mapping strategy names to chunker instances
    """
    chunkers = {}
    
    # Strategy 1: Structural chunking (respects document structure)
    print("Creating structural chunker...")
    structural_config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=config.get("structural", {}).get("target_size", 256),
        respect_headers=True,
        respect_paragraphs=True
    )
    chunkers["structural"] = StructuralChunker(structural_config)
    
    # Strategy 2: Fixed-size chunking (fast baseline)
    print("Creating fixed-size chunker...")
    fixed_config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        target_chunk_size=config.get("fixed_size", {}).get("target_size", 200),
        chunk_overlap=config.get("fixed_size", {}).get("overlap", 20)
    )
    chunkers["fixed_size"] = FixedSizeChunker(fixed_config)
    
    # Strategy 3: Hybrid chunking (combines approaches)
    print("Creating hybrid chunker...")
    hybrid_config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        target_chunk_size=config.get("hybrid", {}).get("target_size", 256),
        use_embeddings=False  # Disable for this example
    )
    chunkers["hybrid"] = HybridChunker(hybrid_config, embedding_service=None)
    
    return chunkers


def process_document(
    chunkers: Dict[str, Any],
    document: str,
    doc_id: str
) -> Dict[str, List[Chunk]]:
    """
    Process document with all chunking strategies and measure performance.
    
    Args:
        chunkers: Dictionary of chunker instances
        document: Document text to process
        doc_id: Document identifier
        
    Returns:
        Dictionary mapping strategy names to chunk lists
    """
    results = {}
    timings = {}
    
    for name, chunker in chunkers.items():
        try:
            print(f"\nProcessing with {name} strategy...")
            start_time = time.time()
            
            chunks = chunker.chunk_document(document, doc_id)
            
            elapsed = time.time() - start_time
            timings[name] = elapsed
            results[name] = chunks
            
            print(f"  ✓ Created {len(chunks)} chunks in {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = []
            timings[name] = 0.0
    
    # Display timing comparison
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    for name, elapsed in timings.items():
        chunks_count = len(results[name])
        if elapsed > 0:
            rate = chunks_count / elapsed
            print(f"{name:15} {elapsed:6.3f}s  {chunks_count:3} chunks  {rate:6.1f} chunks/s")
    
    return results


def display_results(results: Dict[str, List[Chunk]], max_display: int = 2):
    """Display sample results from each strategy."""
    print("\n" + "="*60)
    print("Sample Results")
    print("="*60)
    
    for strategy_name, chunks in results.items():
        if not chunks:
            continue
            
        print(f"\n{strategy_name.upper()} Strategy:")
        print("-" * 40)
        
        for i, chunk in enumerate(chunks[:max_display], 1):
            print(f"\nChunk {i}:")
            print(f"  ID: {chunk.metadata.chunk_id}")
            print(f"  Tokens: {chunk.metadata.token_count}")
            print(f"  Preview: {chunk.text[:100]}...")


def main():
    """Run multi-strategy chunking example."""
    
    # Load configuration
    print("Loading configuration...")
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    
    # Sample document (markdown format)
    document = """# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) combines retrieval with generation to create more accurate and grounded AI responses.

## Key Components

### Vector Database

The vector database stores embeddings of document chunks. It enables semantic search over large document collections using similarity metrics like cosine similarity.

### Embedding Model

Embeddings convert text into dense vectors that capture semantic meaning. Modern embedding models can represent complex relationships between concepts.

### Chunking Strategy

Effective chunking is crucial for RAG performance. The choice of chunking strategy affects both retrieval quality and computational efficiency.

## Benefits

RAG provides several advantages:
- Reduces hallucinations by grounding responses in real data
- Enables knowledge updates without retraining
- Allows citation and verification of sources
- Scales to large knowledge bases

# Implementation Considerations

When implementing RAG systems, consider chunk size, overlap, and retrieval methods carefully.
"""
    
    # Create chunkers
    print("\n" + "="*60)
    print("Initializing Chunking Strategies")
    print("="*60)
    chunkers = create_chunkers(config.get("chunking", {}))
    
    # Process document
    print("\n" + "="*60)
    print("Processing Document")
    print("="*60)
    results = process_document(chunkers, document, "example_doc")
    
    # Display results
    display_results(results)
    
    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for name, chunks in results.items():
        if chunks:
            stats = chunkers[name].get_stats(chunks)
            print(f"\n{name.upper()}:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Avg size: {stats['avg_chunk_size']:.1f} tokens")
            print(f"  Min size: {stats['min_chunk_size']} tokens")
            print(f"  Max size: {stats['max_chunk_size']} tokens")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
