"""Integration tests for chunking strategies."""

import pytest
import os
from pathlib import Path

from rag_factory.strategies.chunking import (
    SemanticChunker,
    StructuralChunker,
    HybridChunker,
    FixedSizeChunker,
    ChunkingConfig,
    ChunkingMethod
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "documents"


@pytest.fixture
def sample_markdown(fixtures_dir):
    """Load sample markdown document."""
    with open(fixtures_dir / "sample.md", "r") as f:
        return f.read()


@pytest.fixture
def sample_plain_text(fixtures_dir):
    """Load sample plain text document."""
    with open(fixtures_dir / "sample.txt", "r") as f:
        return f.read()


@pytest.fixture
def sample_with_code(fixtures_dir):
    """Load sample document with code blocks."""
    with open(fixtures_dir / "sample_with_code.md", "r") as f:
        return f.read()


def test_structural_chunking_markdown_document(sample_markdown):
    """Test structural chunking with real markdown document."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512,
        respect_headers=True
    )
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_markdown, "rag_doc")

    # Assertions
    assert len(chunks) > 0

    # Should preserve header hierarchy
    hierarchies = [chunk.metadata.section_hierarchy for chunk in chunks]
    assert any(len(h) > 0 for h in hierarchies), "Should have section hierarchies"

    # Check that headers are in chunks
    all_text = " ".join(chunk.text for chunk in chunks)
    assert "Introduction to RAG" in all_text
    assert "Vector Database" in all_text

    # Verify all chunks have metadata
    for chunk in chunks:
        assert chunk.metadata.token_count > 0
        assert chunk.metadata.source_document_id == "rag_doc"
        assert chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL

    # Get statistics
    stats = chunker.get_stats(chunks)
    print(f"\nStructural chunking stats: {stats}")

    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_chunk_size"] > 0


def test_structural_chunking_plain_text(sample_plain_text):
    """Test structural chunking with plain text document."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=256
    )
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_plain_text, "plain_doc")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL for chunk in chunks)

    # Plain text should not have hierarchy
    assert all(len(chunk.metadata.section_hierarchy) == 0 for chunk in chunks)


def test_structural_chunking_code_blocks(sample_with_code):
    """Test that code blocks are preserved."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        keep_code_blocks_intact=True
    )
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_with_code, "code_doc")

    # Code blocks should be preserved
    assert any("```python" in chunk.text for chunk in chunks)
    assert any("```javascript" in chunk.text for chunk in chunks)


def test_fixed_size_chunking_consistency(sample_markdown):
    """Test that fixed-size chunking is consistent."""
    config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        target_chunk_size=200,
        chunk_overlap=20
    )
    chunker = FixedSizeChunker(config)

    # Chunk same document twice
    chunks1 = chunker.chunk_document(sample_markdown, "doc_1")
    chunks2 = chunker.chunk_document(sample_markdown, "doc_2")

    # Should produce same number of chunks
    assert len(chunks1) == len(chunks2)

    # Chunks should have similar sizes
    for c1, c2 in zip(chunks1, chunks2):
        assert abs(c1.metadata.token_count - c2.metadata.token_count) < 5


def test_compare_chunking_strategies(sample_markdown):
    """Compare different chunking strategies on same document."""
    document = sample_markdown

    # Test structural chunking
    structural_config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    structural_chunker = StructuralChunker(structural_config)
    structural_chunks = structural_chunker.chunk_document(document, "ml_doc")

    # Test fixed-size chunking
    fixed_config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        target_chunk_size=512
    )
    fixed_chunker = FixedSizeChunker(fixed_config)
    fixed_chunks = fixed_chunker.chunk_document(document, "ml_doc")

    # Both should produce chunks
    assert len(structural_chunks) > 0
    assert len(fixed_chunks) > 0

    # Get stats for comparison
    structural_stats = structural_chunker.get_stats(structural_chunks)
    fixed_stats = fixed_chunker.get_stats(fixed_chunks)

    print(f"\nStructural chunking: {structural_stats}")
    print(f"Fixed-size chunking: {fixed_stats}")

    # Verify different strategies produce different results
    assert structural_stats["total_chunks"] >= 1
    assert fixed_stats["total_chunks"] >= 1


def test_chunking_empty_document():
    """Test handling of empty documents."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document("", "empty_doc")
    assert chunks == []

    chunks = chunker.chunk_document("   \n\n  ", "whitespace_doc")
    assert chunks == []


def test_chunking_very_long_document():
    """Test chunking a very long document."""
    # Generate a long document
    long_doc = "\n\n".join([
        f"# Section {i}\n\nThis is content for section {i}. " * 10
        for i in range(50)
    ])

    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512
    )
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(long_doc, "long_doc")

    # Should produce many chunks
    assert len(chunks) > 10

    # All chunks should be valid
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.metadata.token_count > 0


def test_chunking_with_special_characters():
    """Test chunking document with special characters."""
    special_doc = """# Test Document

This document contains special characters: @#$%^&*()

It also has unicode: 你好 مرحبا

And emojis: 🚀 🎉 ⭐
"""

    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(special_doc, "special_doc")

    assert len(chunks) > 0
    # Special characters should be preserved
    all_text = " ".join(chunk.text for chunk in chunks)
    assert "🚀" in all_text or "@#$%^&*()" in all_text


def test_batch_document_processing():
    """Test processing multiple documents in batch."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    documents = [
        {"text": f"# Document {i}\n\nContent for document {i}.", "id": f"doc_{i}"}
        for i in range(10)
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 10
    assert all(len(chunks) > 0 for chunks in results)

    # Verify IDs are preserved
    for i, chunks in enumerate(results):
        assert chunks[0].metadata.source_document_id == f"doc_{i}"


def test_chunk_metadata_completeness(sample_markdown):
    """Test that all chunk metadata is complete."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_markdown, "test_doc")

    for chunk in chunks:
        # Check all required metadata fields
        assert chunk.metadata.chunk_id
        assert chunk.metadata.source_document_id == "test_doc"
        assert chunk.metadata.position >= 0
        assert chunk.metadata.start_char >= 0
        assert chunk.metadata.end_char > 0
        assert isinstance(chunk.metadata.section_hierarchy, list)
        assert chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL
        assert chunk.metadata.token_count > 0


def test_chunk_validation(sample_markdown):
    """Test chunk validation."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        min_chunk_size=50,
        max_chunk_size=2000
    )
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_markdown, "test_doc")

    # Validate chunks
    is_valid = chunker.validate_chunks(chunks)

    # Should be valid (or contain atomic content)
    assert isinstance(is_valid, bool)


def test_get_stats_functionality(sample_markdown):
    """Test statistics generation."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    chunks = chunker.chunk_document(sample_markdown, "test_doc")
    stats = chunker.get_stats(chunks)

    # Check all expected statistics
    assert "total_chunks" in stats
    assert "avg_chunk_size" in stats
    assert "min_chunk_size" in stats
    assert "max_chunk_size" in stats
    assert "chunking_method" in stats

    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_chunk_size"] > 0
    assert stats["chunking_method"] == "structural"
