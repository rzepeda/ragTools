"""Unit tests for ParentRetriever."""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from rag_factory.strategies.hierarchical.parent_retriever import ParentRetriever
from rag_factory.strategies.hierarchical.models import (
    HierarchicalChunk,
    HierarchyLevel,
    HierarchyMetadata,
    ExpansionStrategy,
    HierarchicalConfig
)


class TestParentRetriever:
    """Tests for ParentRetriever class."""
    
    @pytest.fixture
    def mock_chunk_repo(self):
        """Create a mock chunk repository."""
        return Mock()
    
    @pytest.fixture
    def retriever(self, mock_chunk_repo):
        """Create a ParentRetriever instance."""
        return ParentRetriever(mock_chunk_repo)
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample hierarchical chunk."""
        return HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="This is a paragraph chunk.",
            hierarchy_level=HierarchyLevel.PARAGRAPH,
            hierarchy_metadata=HierarchyMetadata(
                position_in_parent=0,
                total_siblings=3,
                depth_from_root=2
            ),
            parent_chunk_id=str(uuid4()),
            token_count=5
        )
    
    @pytest.fixture
    def parent_chunk(self):
        """Create a sample parent chunk."""
        return HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="This is a section containing multiple paragraphs.",
            hierarchy_level=HierarchyLevel.SECTION,
            hierarchy_metadata=HierarchyMetadata(
                position_in_parent=0,
                total_siblings=2,
                depth_from_root=1
            ),
            parent_chunk_id=None,
            token_count=8
        )
    
    def test_expand_immediate_parent(self, retriever, mock_chunk_repo, sample_chunk, parent_chunk):
        """Test immediate parent expansion strategy."""
        # Setup mock to return parent
        mock_db_chunk = Mock()
        mock_db_chunk.chunk_id = parent_chunk.chunk_id
        mock_db_chunk.document_id = parent_chunk.document_id
        mock_db_chunk.text = parent_chunk.text
        mock_db_chunk.hierarchy_level = parent_chunk.hierarchy_level.value
        mock_db_chunk.hierarchy_metadata = {
            "position_in_parent": 0,
            "total_siblings": 2,
            "depth_from_root": 1
        }
        mock_db_chunk.parent_chunk_id = None
        mock_db_chunk.metadata_ = {}
        
        mock_chunk_repo.get_by_id.return_value = mock_db_chunk
        
        # Expand with immediate parent strategy
        expanded = retriever._expand_immediate_parent(sample_chunk)
        
        assert expanded.original_chunk == sample_chunk
        assert expanded.expansion_strategy == ExpansionStrategy.IMMEDIATE_PARENT
        assert len(expanded.parent_chunks) == 1
        assert parent_chunk.text in expanded.expanded_text
        assert sample_chunk.text in expanded.expanded_text
    
    def test_expand_no_parent(self, retriever, mock_chunk_repo):
        """Test expansion when chunk has no parent."""
        chunk = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="Root chunk",
            hierarchy_level=HierarchyLevel.DOCUMENT,
            hierarchy_metadata=HierarchyMetadata(0, 0, 0),
            parent_chunk_id=None,
            token_count=2
        )
        
        expanded = retriever._expand_immediate_parent(chunk)
        
        assert expanded.expanded_text == chunk.text
        assert len(expanded.parent_chunks) == 0
    
    def test_deduplication(self, retriever, mock_chunk_repo):
        """Test that chunks with same parent are deduplicated."""
        parent_id = str(uuid4())
        
        # Create two chunks with same parent
        chunk1 = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="Chunk 1",
            hierarchy_level=HierarchyLevel.PARAGRAPH,
            hierarchy_metadata=HierarchyMetadata(0, 2, 2),
            parent_chunk_id=parent_id,
            token_count=2
        )
        
        chunk2 = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=chunk1.document_id,
            text="Chunk 2",
            hierarchy_level=HierarchyLevel.PARAGRAPH,
            hierarchy_metadata=HierarchyMetadata(1, 2, 2),
            parent_chunk_id=parent_id,
            token_count=2
        )
        
        # Mock parent
        mock_parent = Mock()
        mock_parent.chunk_id = parent_id
        mock_parent.text = "Parent text"
        mock_parent.hierarchy_level = 1
        mock_parent.hierarchy_metadata = {}
        mock_parent.parent_chunk_id = None
        mock_parent.metadata_ = {}
        
        mock_chunk_repo.get_by_id.return_value = mock_parent
        
        # Expand both chunks
        expanded = retriever.expand_chunks([chunk1, chunk2])
        
        # Should only return one (deduplicated)
        assert len(expanded) == 1
    
    def test_adaptive_strategy_small_chunk(self, retriever, mock_chunk_repo):
        """Test adaptive strategy chooses full section for small chunks."""
        small_chunk = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="Small",
            hierarchy_level=HierarchyLevel.SENTENCE,
            hierarchy_metadata=HierarchyMetadata(0, 1, 3),
            parent_chunk_id=str(uuid4()),
            token_count=1  # Very small
        )
        
        mock_chunk_repo.get_ancestors.return_value = []
        
        # Adaptive should choose full section for small chunks
        expanded = retriever._expand_adaptive(small_chunk)
        
        assert expanded.expansion_strategy == ExpansionStrategy.FULL_SECTION
    
    def test_adaptive_strategy_paragraph_level(self, retriever, mock_chunk_repo, sample_chunk):
        """Test adaptive strategy for paragraph-level chunks."""
        sample_chunk.token_count = 150  # Medium size
        
        mock_chunk_repo.get_by_id.return_value = None
        
        # Adaptive should choose immediate parent for paragraphs
        expanded = retriever._expand_adaptive(sample_chunk)
        
        assert expanded.expansion_strategy == ExpansionStrategy.IMMEDIATE_PARENT
    
    def test_window_expansion(self, retriever, mock_chunk_repo, sample_chunk):
        """Test window expansion with siblings."""
        # Mock parent and siblings
        mock_parent = Mock()
        mock_parent.chunk_id = sample_chunk.parent_chunk_id
        mock_parent.text = "Section text"
        mock_parent.hierarchy_level = 1
        mock_parent.hierarchy_metadata = {}
        mock_parent.parent_chunk_id = None
        mock_parent.metadata_ = {}
        
        # Mock siblings
        sibling1 = Mock()
        sibling1.chunk_id = str(uuid4())
        sibling1.text = "Sibling 1"
        sibling1.hierarchy_level = 2
        sibling1.hierarchy_metadata = {}
        sibling1.parent_chunk_id = sample_chunk.parent_chunk_id
        sibling1.metadata_ = {}
        
        sibling2 = Mock()
        sibling2.chunk_id = sample_chunk.chunk_id
        sibling2.text = sample_chunk.text
        sibling2.hierarchy_level = 2
        sibling2.hierarchy_metadata = {}
        sibling2.parent_chunk_id = sample_chunk.parent_chunk_id
        sibling2.metadata_ = {}
        
        mock_chunk_repo.get_by_id.return_value = mock_parent
        mock_chunk_repo.get_children.return_value = [sibling1, sibling2]
        
        expanded = retriever._expand_window(sample_chunk)
        
        assert expanded.expansion_strategy == ExpansionStrategy.WINDOW
        assert len(expanded.parent_chunks) > 0


class TestParentRetrieverEdgeCases:
    """Test edge cases for ParentRetriever."""
    
    def test_circular_reference_protection(self):
        """Test that circular references don't cause infinite loops."""
        # This would be caught by database validation,
        # but test that retriever handles gracefully
        mock_repo = Mock()
        retriever = ParentRetriever(mock_repo)
        
        # Mock a circular reference scenario
        mock_repo.get_by_id.return_value = None
        
        chunk = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="Chunk",
            hierarchy_level=HierarchyLevel.PARAGRAPH,
            hierarchy_metadata=HierarchyMetadata(0, 1, 2),
            parent_chunk_id=str(uuid4()),
            token_count=1
        )
        
        # Should handle gracefully without infinite loop
        expanded = retriever._expand_immediate_parent(chunk)
        assert expanded is not None
    
    def test_missing_parent_handling(self):
        """Test handling when parent chunk is missing."""
        mock_repo = Mock()
        mock_repo.get_by_id.return_value = None
        
        retriever = ParentRetriever(mock_repo)
        
        chunk = HierarchicalChunk(
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            text="Orphaned chunk",
            hierarchy_level=HierarchyLevel.PARAGRAPH,
            hierarchy_metadata=HierarchyMetadata(0, 1, 2),
            parent_chunk_id=str(uuid4()),
            token_count=2
        )
        
        expanded = retriever._expand_immediate_parent(chunk)
        
        # Should return chunk text without parent
        assert expanded.expanded_text == chunk.text
        assert len(expanded.parent_chunks) == 0
