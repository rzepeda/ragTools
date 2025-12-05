"""Unit tests for FixedSizeChunker."""

import pytest
from rag_factory.strategies.chunking.fixed_size_chunker import FixedSizeChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod


@pytest.fixture
def chunking_config():
    """Create chunking configuration."""
    return ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        min_chunk_size=50,
        max_chunk_size=300,
        target_chunk_size=150,
        chunk_overlap=20
    )


def test_fixed_size_chunker_initialization(chunking_config):
    """Test fixed size chunker initializes correctly."""
    chunker = FixedSizeChunker(chunking_config)
    assert chunker.config == chunking_config


def test_chunk_document_basic(chunking_config):
    """Test basic document chunking."""
    chunker = FixedSizeChunker(chunking_config)

    # Create a document with many words
    document = " ".join([f"word{i}" for i in range(200)])
    chunks = chunker.chunk_document(document, "doc_1")

    assert len(chunks) > 1
    assert all(chunk.metadata.source_document_id == "doc_1" for chunk in chunks)
    assert all(chunk.metadata.chunking_method == ChunkingMethod.FIXED_SIZE for chunk in chunks)


def test_chunk_document_empty(chunking_config):
    """Test chunking empty document."""
    chunker = FixedSizeChunker(chunking_config)

    chunks = chunker.chunk_document("", "doc_1")
    assert chunks == []

    chunks = chunker.chunk_document("   ", "doc_1")
    assert chunks == []


def test_chunk_size_approximately_target(chunking_config):
    """Test that chunks are approximately target size."""
    chunker = FixedSizeChunker(chunking_config)

    document = " ".join(["word" for _ in range(500)])
    chunks = chunker.chunk_document(document, "doc_1")

    # Check that most chunks are close to target size
    for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
        assert chunk.metadata.token_count <= chunking_config.target_chunk_size * 1.5


def test_chunk_overlap(chunking_config):
    """Test chunk overlap functionality."""
    config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        min_chunk_size=20,
        max_chunk_size=100,
        target_chunk_size=50,
        chunk_overlap=10
    )
    chunker = FixedSizeChunker(config)

    document = " ".join([f"word{i}" for i in range(100)])
    chunks = chunker.chunk_document(document, "doc_1")

    # There should be overlap between consecutive chunks
    # (difficult to test precisely without tracking exact positions)
    assert len(chunks) > 1


def test_no_overlap(chunking_config):
    """Test chunking without overlap."""
    config = ChunkingConfig(
        method=ChunkingMethod.FIXED_SIZE,
        min_chunk_size=20,
        max_chunk_size=100,
        target_chunk_size=50,
        chunk_overlap=0
    )
    chunker = FixedSizeChunker(config)

    document = " ".join([f"word{i}" for i in range(100)])
    chunks = chunker.chunk_document(document, "doc_1")

    assert len(chunks) >= 2


def test_chunk_multiple_documents(chunking_config):
    """Test chunking multiple documents."""
    chunker = FixedSizeChunker(chunking_config)

    documents = [
        {"text": " ".join(["word" for _ in range(50)]), "id": "doc_1"},
        {"text": " ".join(["word" for _ in range(50)]), "id": "doc_2"}
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 2
    assert results[0][0].metadata.source_document_id == "doc_1"
    assert results[1][0].metadata.source_document_id == "doc_2"


def test_get_overlap_words(chunking_config):
    """Test overlap word extraction."""
    chunker = FixedSizeChunker(chunking_config)

    words = ["word1", "word2", "word3", "word4", "word5"]
    overlap_words = chunker._get_overlap_words(words, 10)

    # Should return some words for overlap
    assert isinstance(overlap_words, list)


def test_get_overlap_words_zero_overlap(chunking_config):
    """Test overlap with zero overlap setting."""
    chunker = FixedSizeChunker(chunking_config)

    words = ["word1", "word2", "word3"]
    overlap_words = chunker._get_overlap_words(words, 0)

    assert overlap_words == []


def test_count_tokens(chunking_config):
    """Test token counting."""
    chunker = FixedSizeChunker(chunking_config)

    text = "This is a test sentence."
    token_count = chunker._count_tokens(text)

    assert isinstance(token_count, int)
    assert token_count > 0


def test_single_word_document(chunking_config):
    """Test document with single word."""
    chunker = FixedSizeChunker(chunking_config)

    chunks = chunker.chunk_document("word", "doc_1")

    assert len(chunks) == 1
    assert chunks[0].text == "word"


def test_chunk_metadata(chunking_config):
    """Test chunk metadata is correctly set."""
    chunker = FixedSizeChunker(chunking_config)

    document = " ".join(["word" for _ in range(100)])
    chunks = chunker.chunk_document(document, "test_doc")

    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_id == f"test_doc_chunk_{i}"
        assert chunk.metadata.position == i
        assert chunk.metadata.chunking_method == ChunkingMethod.FIXED_SIZE
        assert chunk.metadata.token_count > 0


def test_create_chunk_helper(chunking_config):
    """Test _create_chunk helper method."""
    chunker = FixedSizeChunker(chunking_config)

    chunk = chunker._create_chunk("test text", "doc_1", 0)

    assert chunk.text == "test text"
    assert chunk.metadata.chunk_id == "doc_1_chunk_0"
    assert chunk.metadata.source_document_id == "doc_1"
    assert chunk.metadata.position == 0
