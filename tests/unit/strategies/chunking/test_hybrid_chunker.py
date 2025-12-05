"""Unit tests for HybridChunker."""

import pytest
from unittest.mock import Mock
from rag_factory.strategies.chunking.hybrid_chunker import HybridChunker
from rag_factory.strategies.chunking.base import (
    ChunkingConfig,
    ChunkingMethod,
    Chunk,
    ChunkMetadata
)


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock()
    service.embed.return_value = Mock(
        embeddings=[[0.5, 0.5, 0.5] for _ in range(10)]
    )
    return service


@pytest.fixture
def chunking_config():
    """Create chunking configuration."""
    return ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        target_chunk_size=512,
        use_embeddings=True
    )


def test_hybrid_chunker_initialization_with_embeddings(chunking_config, mock_embedding_service):
    """Test hybrid chunker initializes with embeddings."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    assert chunker.config == chunking_config
    assert chunker.use_semantic is True
    assert chunker.semantic_chunker is not None
    assert chunker.structural_chunker is not None


def test_hybrid_chunker_initialization_without_embeddings():
    """Test hybrid chunker without embeddings (structural only)."""
    config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        use_embeddings=False
    )
    chunker = HybridChunker(config, None)

    assert chunker.use_semantic is False
    assert chunker.semantic_chunker is None
    assert chunker.structural_chunker is not None


def test_chunk_markdown_document(chunking_config, mock_embedding_service):
    """Test chunking markdown document."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    markdown = """# Header 1

Content under header 1.

## Header 1.1

Content under header 1.1.

# Header 2

Content under header 2.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.HYBRID for chunk in chunks)


def test_chunk_document_empty(chunking_config, mock_embedding_service):
    """Test chunking empty document."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    chunks = chunker.chunk_document("", "doc_1")
    assert chunks == []


def test_structural_only_mode():
    """Test hybrid chunker in structural-only mode."""
    config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        use_embeddings=False
    )
    chunker = HybridChunker(config, None)

    markdown = """# Header

Content here."""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    # Should still mark as HYBRID even in structural-only mode
    assert all(chunk.metadata.chunking_method == ChunkingMethod.HYBRID for chunk in chunks)


def test_semantic_refinement_of_large_chunks(chunking_config, mock_embedding_service):
    """Test that large chunks are refined with semantic chunking."""
    config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        min_chunk_size=20,
        target_chunk_size=50,  # Small size to trigger refinement
        max_chunk_size=200,
        use_embeddings=True
    )
    chunker = HybridChunker(config, mock_embedding_service)

    # Create a large document
    document = " ".join([f"Sentence {i}." for i in range(100)])
    chunks = chunker.chunk_document(document, "doc_1")

    assert len(chunks) > 0


def test_hierarchy_preservation(chunking_config, mock_embedding_service):
    """Test that section hierarchy is preserved."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    markdown = """# Main Section

## Subsection

Content with multiple sentences that might be split semantically.
More content here.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    # Check that some chunks have hierarchy
    chunks_with_hierarchy = [c for c in chunks if c.metadata.section_hierarchy]
    assert len(chunks_with_hierarchy) > 0


def test_chunk_multiple_documents(chunking_config, mock_embedding_service):
    """Test chunking multiple documents."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    documents = [
        {"text": "# Doc 1\nContent", "id": "doc_1"},
        {"text": "# Doc 2\nContent", "id": "doc_2"}
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 2
    assert results[0][0].metadata.source_document_id == "doc_1"
    assert results[1][0].metadata.source_document_id == "doc_2"


def test_atomic_content_not_split(chunking_config, mock_embedding_service):
    """Test that atomic content (code blocks) is not split."""
    config = ChunkingConfig(
        method=ChunkingMethod.HYBRID,
        min_chunk_size=5,
        target_chunk_size=10,  # Very small to force splitting
        max_chunk_size=50,
        use_embeddings=True,
        keep_code_blocks_intact=True
    )
    chunker = HybridChunker(config, mock_embedding_service)

    code = """```python
def long_function():
    return "This is a long code block that exceeds the max chunk size"
```"""

    chunks = chunker.chunk_document(code, "doc_1")

    # Code should be kept as single chunk
    assert any("```python" in chunk.text for chunk in chunks)


def test_get_stats(chunking_config, mock_embedding_service):
    """Test getting statistics about chunks."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    markdown = """# Header

Content here.

## Subheader

More content."""

    chunks = chunker.chunk_document(markdown, "doc_1")
    stats = chunker.get_stats(chunks)

    assert "total_chunks" in stats
    assert "avg_chunk_size" in stats
    assert "chunks_with_hierarchy" in stats
    assert "semantic_refinement_enabled" in stats
    assert stats["semantic_refinement_enabled"] is True


def test_chunk_position_reindexing(chunking_config, mock_embedding_service):
    """Test that chunks are correctly reindexed."""
    chunker = HybridChunker(chunking_config, mock_embedding_service)

    document = """# Section 1
Content 1

# Section 2
Content 2"""

    chunks = chunker.chunk_document(document, "doc_1")

    # Check that positions are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.position == i


def test_semantic_chunking_failure_fallback(chunking_config):
    """Test fallback when semantic chunking fails."""
    # Create a mock that raises an error
    mock_service = Mock()
    mock_service.embed.side_effect = Exception("Embedding failed")

    chunker = HybridChunker(chunking_config, mock_service)

    # Create large section that would trigger semantic chunking
    document = " ".join([f"Sentence {i}." for i in range(200)])
    chunks = chunker.chunk_document(document, "doc_1")

    # Should still return chunks (fallback to structural)
    assert len(chunks) > 0
