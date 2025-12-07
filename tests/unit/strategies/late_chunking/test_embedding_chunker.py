"""
Unit tests for embedding chunker.
"""

import pytest
import numpy as np

from rag_factory.strategies.late_chunking.embedding_chunker import EmbeddingChunker
from rag_factory.strategies.late_chunking.models import (
    DocumentEmbedding,
    TokenEmbedding,
    LateChunkingConfig,
    EmbeddingChunkingMethod
)


@pytest.fixture
def chunker_config():
    """Create test configuration for chunker."""
    return LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.FIXED_SIZE,
        target_chunk_size=10,
        min_chunk_size=5,
        max_chunk_size=20,
        chunk_overlap_tokens=2
    )


@pytest.fixture
def embedding_chunker(chunker_config):
    """Create embedding chunker instance."""
    return EmbeddingChunker(chunker_config)


@pytest.fixture
def sample_doc_embedding():
    """Create sample DocumentEmbedding for testing."""
    # Create fake token embeddings
    tokens = []
    text = "This is a test document with many tokens for chunking"

    words = text.split()
    char_pos = 0

    for i, word in enumerate(words):
        token = TokenEmbedding(
            token=word,
            token_id=i,
            start_char=char_pos,
            end_char=char_pos + len(word),
            embedding=np.random.randn(384).tolist(),
            position=i
        )
        tokens.append(token)
        char_pos += len(word) + 1  # +1 for space

    doc_emb = DocumentEmbedding(
        document_id="test_doc",
        text=text,
        full_embedding=np.random.randn(384).tolist(),
        token_embeddings=tokens,
        model_name="test_model",
        token_count=len(tokens),
        embedding_dim=384
    )

    return doc_emb


def test_fixed_size_chunking(embedding_chunker, sample_doc_embedding):
    """Test fixed-size chunking."""
    chunks = embedding_chunker._fixed_size_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    # All chunks should have reasonable size
    for chunk in chunks:
        assert chunk.token_count <= embedding_chunker.config.max_chunk_size
        assert chunk.document_id == sample_doc_embedding.document_id
        assert chunk.chunking_method == EmbeddingChunkingMethod.FIXED_SIZE


def test_semantic_boundary_chunking(sample_doc_embedding):
    """Test semantic boundary-based chunking."""
    config = LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.SEMANTIC_BOUNDARY,
        similarity_threshold=0.5,
        min_chunk_size=3
    )
    chunker = EmbeddingChunker(config)

    chunks = chunker._semantic_boundary_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    # Chunks should respect boundaries
    for chunk in chunks:
        assert chunk.token_count >= config.min_chunk_size or len(chunks) == 1
        assert chunk.chunking_method == EmbeddingChunkingMethod.SEMANTIC_BOUNDARY


def test_adaptive_chunking(sample_doc_embedding):
    """Test adaptive chunking based on variance."""
    config = LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.ADAPTIVE,
        min_chunk_size=3,
        max_chunk_size=15
    )
    chunker = EmbeddingChunker(config)

    chunks = chunker._adaptive_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunking_method == EmbeddingChunkingMethod.ADAPTIVE
        assert chunk.token_count <= config.max_chunk_size


def test_chunk_text_reconstruction(embedding_chunker, sample_doc_embedding):
    """Test that chunk text is correctly reconstructed."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    for chunk in chunks:
        # Text should match character range
        expected_text = sample_doc_embedding.text[chunk.char_range[0]:chunk.char_range[1]]
        assert chunk.text == expected_text


def test_chunk_embedding_averaging(embedding_chunker, sample_doc_embedding):
    """Test that chunk embeddings are averaged correctly."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    for chunk in chunks:
        # Chunk embedding should be average of token embeddings
        start_token_idx = chunk.token_range[0]
        end_token_idx = chunk.token_range[1]

        # Find tokens in this range
        chunk_tokens = [
            t for t in sample_doc_embedding.token_embeddings
            if start_token_idx <= t.position <= end_token_idx
        ]

        if chunk_tokens:
            token_embeddings = [t.embedding for t in chunk_tokens]
            expected_avg = np.mean(token_embeddings, axis=0)
            actual_avg = np.array(chunk.chunk_embedding)

            assert np.allclose(actual_avg, expected_avg, atol=1e-5)


def test_cosine_similarity(embedding_chunker):
    """Test cosine similarity calculation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])

    similarity = embedding_chunker._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0

    vec3 = np.array([0.0, 1.0, 0.0])
    similarity = embedding_chunker._cosine_similarity(vec1, vec3)
    assert abs(similarity - 0.0) < 1e-6  # Should be exactly 0.0


def test_cosine_similarity_zero_vectors(embedding_chunker):
    """Test cosine similarity with zero vectors."""
    vec1 = np.array([0.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])

    similarity = embedding_chunker._cosine_similarity(vec1, vec2)
    assert similarity == 0.0


def test_chunk_overlap(sample_doc_embedding):
    """Test that chunk overlap is working."""
    config = LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.FIXED_SIZE,
        target_chunk_size=5,
        chunk_overlap_tokens=2
    )
    chunker = EmbeddingChunker(config)

    chunks = chunker._fixed_size_chunking(sample_doc_embedding)

    # Check that consecutive chunks have overlap
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].token_range[1]
            chunk2_start = chunks[i + 1].token_range[0]
            # There should be overlap
            assert chunk1_end >= chunk2_start


def test_chunk_ids_unique(embedding_chunker, sample_doc_embedding):
    """Test that chunk IDs are unique."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # All unique


def test_detect_semantic_boundaries(embedding_chunker):
    """Test semantic boundary detection."""
    # Create tokens with varying similarity
    tokens = []
    for i in range(20):
        # Create embeddings that have clear boundaries
        if i < 10:
            embedding = [1.0] * 384  # Similar embeddings
        else:
            embedding = [-1.0] * 384  # Different embeddings

        token = TokenEmbedding(
            token=f"token{i}",
            token_id=i,
            start_char=i * 6,
            end_char=(i + 1) * 6,
            embedding=embedding,
            position=i
        )
        tokens.append(token)

    boundaries = embedding_chunker._detect_semantic_boundaries(tokens)

    assert 0 in boundaries  # Start boundary
    assert len(tokens) in boundaries  # End boundary
    assert len(boundaries) >= 2  # At least start and end


def test_hierarchical_chunking_fallback(embedding_chunker, sample_doc_embedding):
    """Test that hierarchical chunking falls back to semantic boundary."""
    chunks = embedding_chunker._hierarchical_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    # Should use semantic boundary as fallback
    for chunk in chunks:
        assert chunk.chunking_method == EmbeddingChunkingMethod.SEMANTIC_BOUNDARY


def test_chunk_metadata(embedding_chunker, sample_doc_embedding):
    """Test that chunk metadata is properly set."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    for chunk in chunks:
        assert chunk.chunk_id is not None
        assert chunk.document_id == sample_doc_embedding.document_id
        assert chunk.token_range is not None
        assert chunk.char_range is not None
        assert chunk.token_count > 0
        assert len(chunk.chunk_embedding) == sample_doc_embedding.embedding_dim
