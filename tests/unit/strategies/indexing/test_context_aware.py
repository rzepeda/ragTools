"""Unit tests for ContextAwareChunkingIndexing strategy."""

import pytest
from unittest.mock import Mock, AsyncMock
import numpy as np

from rag_factory.strategies.indexing.context_aware import ContextAwareChunkingIndexing
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies
from rag_factory.core.indexing_interface import IndexingContext

@pytest.fixture
def mock_deps():
    deps = Mock(spec=StrategyDependencies)
    deps.embedding_service = Mock()
    deps.database_service = Mock()
    deps.validate_for_strategy.return_value = (True, [])
    return deps

@pytest.fixture
def strategy(mock_deps):
    config = {
        "chunk_size_min": 10,
        "chunk_size_max": 50,
        "chunk_size_target": 30,
        "boundary_threshold": 0.8,
        "window_size": 1
    }
    return ContextAwareChunkingIndexing(config, mock_deps)

def test_capabilities(strategy):
    """Test produces capabilities."""
    assert strategy.produces() == {
        IndexCapability.CHUNKS,
        IndexCapability.VECTORS,
        IndexCapability.DATABASE
    }

def test_requirements(strategy):
    """Test service requirements."""
    assert strategy.requires_services() == {
        ServiceDependency.EMBEDDING,
        ServiceDependency.DATABASE
    }

def test_split_into_sentences(strategy):
    """Test sentence splitting."""
    text = "Hello world. This is a test! Another one?"
    sentences = strategy._split_into_sentences(text)
    assert sentences == ["Hello world.", "This is a test!", "Another one?"]

def test_create_windows(strategy):
    """Test window creation."""
    sentences = ["S1", "S2", "S3", "S4"]
    # Window size 1 means [i-1, i, i+2] -> 3 sentences context centered roughly
    # The implementation uses: start = max(0, i - 1), end = min(len, i + 2)
    # i=0: 0:2 -> S1 S2
    # i=1: 0:3 -> S1 S2 S3
    # i=2: 1:4 -> S2 S3 S4
    # i=3: 2:4 -> S3 S4
    
    windows = strategy._create_windows(sentences, window_size=1)
    assert len(windows) == 4
    assert windows[0] == "S1 S2"
    assert windows[1] == "S1 S2 S3"
    assert windows[2] == "S2 S3 S4"
    assert windows[3] == "S3 S4"

def test_find_boundaries(strategy):
    """Test boundary detection."""
    # Create embeddings with clear similarity pattern
    # v1, v2 similar; v3 different
    v1 = [1.0, 0.0]
    v2 = [0.9, 0.1]  # High sim with v1
    v3 = [0.0, 1.0]  # Low sim with v2
    
    embeddings = [v1, v2, v3]
    
    # Mock _cosine_similarity to avoid using mocked numpy
    # sim(v1, v2) ~ 0.9
    # sim(v2, v3) ~ 0.1
    strategy._cosine_similarity = Mock(side_effect=[0.9, 0.1])
    
    # Threshold 0.5
    # sim(v1, v2) = 0.9 > 0.5 -> No boundary
    # sim(v2, v3) = 0.1 < 0.5 -> Boundary at index 1 (between v2 and v3)
    
    boundaries = strategy._find_boundaries(embeddings, threshold=0.5)
    assert boundaries == [1]

def test_create_chunks_respects_boundaries(strategy):
    """Test chunk creation respects boundaries."""
    sentences = ["S1", "S2", "S3", "S4"]
    # Boundary at 1 means split after S2 (index 1)
    # S1, S2 | S3, S4
    boundaries = [1]
    
    # Config allows small chunks for this test
    chunks = strategy._create_chunks(
        sentences,
        boundaries,
        min_size=1,
        max_size=100
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == "S1 S2"
    assert chunks[1]["text"] == "S3 S4"

def test_create_chunks_respects_min_size(strategy):
    """Test chunk creation ignores boundary if min size not met."""
    sentences = ["Short", "Short", "Longer sentence here"]
    # Boundary at 0 (after first "Short")
    boundaries = [0]
    
    # Min size large enough to force merge
    chunks = strategy._create_chunks(
        sentences,
        boundaries,
        min_size=15, # "Short" is 5 chars. "Short Short" is 11. 
        max_size=100
    )
    
    # Should merge despite boundary because "Short" < 15
    # Actually logic is: accumulate. At boundary, check if current_len >= min_size.
    # "Short" (5) < 15 -> Don't split.
    # Next: "Short" (5). Total "Short Short" (11).
    # Next: "Longer..."
    
    assert len(chunks) == 1
    assert "Short Short" in chunks[0]["text"]

def test_create_chunks_respects_max_size(strategy):
    """Test chunk creation forces split at max size."""
    sentences = ["S1", "S2", "S3"]
    boundaries = [] # No semantic boundaries
    
    # Max size small enough to force split
    # "S1" is 2 chars. "S1 S2" is 5 chars.
    chunks = strategy._create_chunks(
        sentences,
        boundaries,
        min_size=1,
        max_size=4 # "S1 S2" (5) > 4
    )
    
    assert len(chunks) == 3
    assert chunks[0]["text"] == "S1"
    assert chunks[1]["text"] == "S2"
    assert chunks[2]["text"] == "S3"

@pytest.mark.asyncio
async def test_process_flow(strategy, mock_deps):
    """Test full process flow."""
    documents = [{"id": "doc1", "text": "Sentence 1. Sentence 2. Sentence 3."}]
    context = Mock(spec=IndexingContext)
    context.database = Mock()
    context.database.store_chunks = AsyncMock()
    
    # Mock embedding service
    # 3 sentences -> 3 windows
    mock_deps.embedding_service.embed_batch = AsyncMock(return_value=[
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]
    ])
    
    result = await strategy.process(documents, context)
    
    assert result.document_count == 1
    assert result.chunk_count > 0
    
    # Verify storage call
    context.database.store_chunks.assert_called_once()
    stored_chunks = context.database.store_chunks.call_args[0][0]
    assert len(stored_chunks) > 0
    assert stored_chunks[0]["metadata"]["strategy"] == "context_aware"
    assert stored_chunks[0]["metadata"]["document_id"] == "doc1"
