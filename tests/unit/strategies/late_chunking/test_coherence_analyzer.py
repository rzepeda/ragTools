"""
Unit tests for coherence analyzer.
"""

import pytest
import numpy as np

from rag_factory.strategies.late_chunking.coherence_analyzer import CoherenceAnalyzer
from rag_factory.strategies.late_chunking.models import (
    EmbeddingChunk,
    EmbeddingChunkingMethod,
    LateChunkingConfig
)


@pytest.fixture
def analyzer_config():
    """Create test configuration for analyzer."""
    return LateChunkingConfig(
        coherence_window_size=3,
        target_chunk_size=100
    )


@pytest.fixture
def coherence_analyzer(analyzer_config):
    """Create coherence analyzer instance."""
    return CoherenceAnalyzer(analyzer_config)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = []
    for i in range(5):
        chunk = EmbeddingChunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            text=f"This is chunk {i} with some text",
            chunk_embedding=np.random.randn(384).tolist(),
            token_range=(i * 10, (i + 1) * 10),
            char_range=(i * 50, (i + 1) * 50),
            token_count=10,
            chunking_method=EmbeddingChunkingMethod.FIXED_SIZE
        )
        chunks.append(chunk)
    return chunks


def test_analyze_chunk_coherence(coherence_analyzer, sample_chunks):
    """Test coherence analysis for chunks."""
    analyzed_chunks = coherence_analyzer.analyze_chunk_coherence(sample_chunks)

    assert len(analyzed_chunks) == len(sample_chunks)
    
    # All chunks should have coherence scores
    for chunk in analyzed_chunks:
        assert chunk.coherence_score is not None
        assert 0.0 <= chunk.coherence_score <= 1.0


def test_coherence_score_calculation(coherence_analyzer):
    """Test coherence score calculation."""
    chunk = EmbeddingChunk(
        chunk_id="test_chunk",
        document_id="test_doc",
        text="Test chunk text",
        chunk_embedding=np.random.randn(384).tolist(),
        token_range=(0, 100),
        char_range=(0, 500),
        token_count=100,
        chunking_method=EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
    )

    score = coherence_analyzer._calculate_intra_chunk_coherence(chunk)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_coherence_score_size_penalty(coherence_analyzer):
    """Test that coherence score considers chunk size."""
    # Chunk at target size should have higher score
    chunk_optimal = EmbeddingChunk(
        chunk_id="optimal",
        document_id="test_doc",
        text="Optimal size chunk",
        chunk_embedding=np.random.randn(384).tolist(),
        token_range=(0, 100),
        char_range=(0, 500),
        token_count=100,  # At target size
        chunking_method=EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
    )

    # Chunk far from target size
    chunk_suboptimal = EmbeddingChunk(
        chunk_id="suboptimal",
        document_id="test_doc",
        text="Small chunk",
        chunk_embedding=np.random.randn(384).tolist(),
        token_range=(0, 10),
        char_range=(0, 50),
        token_count=10,  # Much smaller than target
        chunking_method=EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
    )

    score_optimal = coherence_analyzer._calculate_intra_chunk_coherence(chunk_optimal)
    score_suboptimal = coherence_analyzer._calculate_intra_chunk_coherence(chunk_suboptimal)

    # Optimal size should have higher coherence
    assert score_optimal >= score_suboptimal


def test_compare_with_traditional(coherence_analyzer, sample_chunks):
    """Test comparison with traditional chunking."""
    # Create fake traditional chunks
    traditional_chunks = [
        {"text": f"Traditional chunk {i}", "token_count": 15}
        for i in range(4)
    ]

    # Add coherence scores to late chunks
    for chunk in sample_chunks:
        chunk.coherence_score = 0.85

    metrics = coherence_analyzer.compare_with_traditional(
        sample_chunks,
        traditional_chunks
    )

    assert "late_chunking_coherence" in metrics
    assert "num_late_chunks" in metrics
    assert "num_traditional_chunks" in metrics
    assert "avg_late_chunk_size" in metrics

    assert metrics["num_late_chunks"] == len(sample_chunks)
    assert metrics["num_traditional_chunks"] == len(traditional_chunks)
    assert metrics["late_chunking_coherence"] == 0.85


def test_compare_metrics_types(coherence_analyzer, sample_chunks):
    """Test that comparison metrics have correct types."""
    traditional_chunks = [{"text": "chunk", "token_count": 10}]

    for chunk in sample_chunks:
        chunk.coherence_score = 0.8

    metrics = coherence_analyzer.compare_with_traditional(
        sample_chunks,
        traditional_chunks
    )

    assert isinstance(metrics["late_chunking_coherence"], float)
    assert isinstance(metrics["num_late_chunks"], int)
    assert isinstance(metrics["num_traditional_chunks"], int)
    assert isinstance(metrics["avg_late_chunk_size"], float)


def test_coherence_with_no_scores(coherence_analyzer):
    """Test comparison when chunks have no coherence scores."""
    chunks_no_scores = [
        EmbeddingChunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            text=f"Chunk {i}",
            chunk_embedding=np.random.randn(384).tolist(),
            token_range=(i * 10, (i + 1) * 10),
            char_range=(i * 50, (i + 1) * 50),
            token_count=10,
            chunking_method=EmbeddingChunkingMethod.FIXED_SIZE,
            coherence_score=None
        )
        for i in range(3)
    ]

    traditional_chunks = [{"text": "chunk", "token_count": 10}]

    metrics = coherence_analyzer.compare_with_traditional(
        chunks_no_scores,
        traditional_chunks
    )

    # Should handle missing scores gracefully
    assert metrics["late_chunking_coherence"] == 0.0


def test_analyze_preserves_chunk_data(coherence_analyzer, sample_chunks):
    """Test that analysis preserves original chunk data."""
    original_ids = [c.chunk_id for c in sample_chunks]
    original_texts = [c.text for c in sample_chunks]

    analyzed_chunks = coherence_analyzer.analyze_chunk_coherence(sample_chunks)

    # Original data should be preserved
    for i, chunk in enumerate(analyzed_chunks):
        assert chunk.chunk_id == original_ids[i]
        assert chunk.text == original_texts[i]
