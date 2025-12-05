"""Unit tests for SemanticChunker."""

import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.chunking.semantic_chunker import SemanticChunker
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
    # Mock embedding result with similar vectors
    service.embed.return_value = Mock(
        embeddings=[[0.5, 0.5, 0.5] for _ in range(10)]
    )
    return service


@pytest.fixture
def chunking_config():
    """Create chunking configuration."""
    return ChunkingConfig(
        method=ChunkingMethod.SEMANTIC,
        target_chunk_size=512,
        min_chunk_size=128,
        max_chunk_size=1024,
        similarity_threshold=0.7,
        use_embeddings=True
    )


def test_semantic_chunker_initialization(chunking_config, mock_embedding_service):
    """Test semantic chunker initializes correctly."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)
    assert chunker.config == chunking_config
    assert chunker.embedding_service == mock_embedding_service


def test_semantic_chunker_requires_embeddings():
    """Test that semantic chunker requires embeddings enabled."""
    config = ChunkingConfig(use_embeddings=False)
    with pytest.raises(ValueError, match="requires use_embeddings=True"):
        SemanticChunker(config, Mock())


def test_chunk_document_basic(chunking_config, mock_embedding_service):
    """Test basic document chunking."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    document = "This is the first sentence. This is the second sentence. This is the third sentence."
    chunks = chunker.chunk_document(document, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.source_document_id == "doc_1" for chunk in chunks)
    assert all(chunk.metadata.chunking_method == ChunkingMethod.SEMANTIC for chunk in chunks)


def test_chunk_document_empty():
    """Test chunking empty document."""
    config = ChunkingConfig(use_embeddings=True)
    chunker = SemanticChunker(config, Mock())

    chunks = chunker.chunk_document("", "doc_1")
    assert chunks == []

    chunks = chunker.chunk_document("   ", "doc_1")
    assert chunks == []


def test_split_into_sentences(chunking_config, mock_embedding_service):
    """Test sentence splitting."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    text = "First sentence. Second sentence! Third sentence?"
    sentences = chunker._split_into_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"


def test_cosine_similarity(chunking_config, mock_embedding_service):
    """Test cosine similarity calculation."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    # Identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = chunker._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0)

    # Orthogonal vectors
    vec3 = [0.0, 1.0, 0.0]
    similarity = chunker._cosine_similarity(vec1, vec3)
    assert similarity == pytest.approx(0.0)


def test_detect_boundaries_high_similarity(chunking_config, mock_embedding_service):
    """Test boundary detection with high similarity."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    # All embeddings identical (high similarity)
    embeddings = [[0.5, 0.5] for _ in range(5)]
    boundaries = chunker._detect_boundaries(embeddings, 5)

    # Should only have start and end boundaries
    assert 0 in boundaries
    assert 5 in boundaries


def test_detect_boundaries_low_similarity(chunking_config, mock_embedding_service):
    """Test boundary detection with low similarity."""
    config = ChunkingConfig(similarity_threshold=0.9, use_embeddings=True)
    chunker = SemanticChunker(config, mock_embedding_service)

    # Embeddings with low similarity
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],  # Low similarity -> boundary
        [0.0, 0.9],
        [0.1, 0.9]
    ]

    boundaries = chunker._detect_boundaries(embeddings, 4)

    # Should detect boundary where similarity drops
    assert len(boundaries) > 2  # More than just start and end


def test_single_sentence_document(chunking_config, mock_embedding_service):
    """Test document with single sentence."""
    mock_embedding_service.embed.return_value = Mock(embeddings=[[0.5] * 10])

    chunker = SemanticChunker(chunking_config, mock_embedding_service)
    chunks = chunker.chunk_document("Single sentence.", "doc_1")

    assert len(chunks) == 1
    assert chunks[0].text == "Single sentence."


def test_fallback_chunking(chunking_config, mock_embedding_service):
    """Test fallback chunking when embeddings fail."""
    # Make embedding service raise an error
    mock_embedding_service.embed.side_effect = Exception("Embedding failed")

    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    document = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunker.chunk_document(document, "doc_1")

    # Should still return chunks via fallback
    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.SEMANTIC for chunk in chunks)


def test_chunk_multiple_documents(chunking_config, mock_embedding_service):
    """Test chunking multiple documents."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    documents = [
        {"text": "First document. With sentences.", "id": "doc_1"},
        {"text": "Second document. Also with sentences.", "id": "doc_2"}
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 2
    assert all(isinstance(chunks, list) for chunks in results)
    assert results[0][0].metadata.source_document_id == "doc_1"
    assert results[1][0].metadata.source_document_id == "doc_2"


def test_count_tokens_with_tokenizer(chunking_config, mock_embedding_service):
    """Test token counting with tiktoken."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    text = "This is a test sentence."
    token_count = chunker._count_tokens(text)

    # Should return a positive integer
    assert isinstance(token_count, int)
    assert token_count > 0


def test_create_segments(chunking_config, mock_embedding_service):
    """Test segment creation for embedding."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    sentences = ["First.", "Second.", "Third.", "Fourth.", "Fifth."]
    segments = chunker._create_segments(sentences, segment_size=3)

    # Should create overlapping segments
    assert len(segments) >= len(sentences) - 2
    assert segments[0] == ["First.", "Second.", "Third."]
    assert segments[1] == ["Second.", "Third.", "Fourth."]


def test_coherence_score_computation():
    """Test coherence score calculation."""
    config = ChunkingConfig(
        use_embeddings=True,
        compute_coherence_scores=True
    )

    # Mock service that returns different embeddings for sentences
    mock_service = Mock()
    mock_service.embed.return_value = Mock(
        embeddings=[[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]]
    )

    chunker = SemanticChunker(config, mock_service)

    # Create a test chunk
    chunk = Chunk(
        text="First sentence. Second sentence. Third sentence.",
        metadata=ChunkMetadata(
            chunk_id="c1",
            source_document_id="doc1",
            position=0,
            start_char=0,
            end_char=48,
            section_hierarchy=[],
            chunking_method=ChunkingMethod.SEMANTIC,
            token_count=10
        )
    )

    result = chunker._compute_coherence_scores([chunk])

    # Should have coherence score
    assert result[0].metadata.coherence_score is not None
    assert 0.0 <= result[0].metadata.coherence_score <= 1.0
