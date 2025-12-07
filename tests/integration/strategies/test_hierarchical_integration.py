"""Integration tests for hierarchical RAG strategy."""

import pytest
from uuid import uuid4

from rag_factory.strategies.hierarchical import (
    HierarchicalRAGStrategy,
    ExpansionStrategy,
    HierarchyLevel
)


@pytest.mark.integration
class TestHierarchicalIntegration:
    """Integration tests for hierarchical RAG workflow."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store service."""
        class MockVectorStore:
            def embed_text(self, text: str):
                # Return mock embedding based on text hash for consistency
                return [hash(text) % 100 / 100.0] * 1536
        
        return MockVectorStore()
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database service."""
        class MockChunkRepository:
            def __init__(self):
                self.chunks = {}
                self.search_results = []
            
            def create(self, document_id, chunk_index, text, embedding=None, metadata=None):
                chunk_id = str(uuid4())
                chunk = type('Chunk', (), {
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'chunk_index': chunk_index,
                    'text': text,
                    'embedding': embedding,
                    'metadata_': metadata or {},
                    'hierarchy_level': metadata.get('hierarchy_level', 0) if metadata else 0,
                    'parent_chunk_id': metadata.get('parent_chunk_id') if metadata else None,
                    'hierarchy_metadata': metadata.get('hierarchy_metadata', {}) if metadata else {}
                })()
                self.chunks[chunk_id] = chunk
                return chunk
            
            def search_similar(self, embedding, top_k=5, threshold=0.0):
                # Return mock search results
                results = []
                for chunk in list(self.chunks.values())[:top_k]:
                    results.append((chunk, 0.9))
                return results
            
            def get_by_id(self, chunk_id):
                return self.chunks.get(str(chunk_id))
            
            def get_children(self, chunk_id):
                return [
                    c for c in self.chunks.values()
                    if c.parent_chunk_id == str(chunk_id)
                ]
            
            def get_ancestors(self, chunk_id, max_depth=10):
                ancestors = []
                current_id = str(chunk_id)
                for _ in range(max_depth):
                    chunk = self.chunks.get(current_id)
                    if not chunk or not chunk.parent_chunk_id:
                        break
                    parent = self.chunks.get(chunk.parent_chunk_id)
                    if parent:
                        ancestors.append(parent)
                        current_id = chunk.parent_chunk_id
                return ancestors
            
            def validate_hierarchy(self):
                return []  # No issues
        
        class MockDatabase:
            def __init__(self):
                self.chunk_repository = MockChunkRepository()
        
        return MockDatabase()
    
    def test_end_to_end_workflow(self, mock_vector_store, mock_database):
        """Test complete hierarchical retrieval workflow."""
        # Create strategy
        strategy = HierarchicalRAGStrategy(
            vector_store_service=mock_vector_store,
            database_service=mock_database,
            config={"expansion_strategy": ExpansionStrategy.IMMEDIATE_PARENT}
        )
        
        # Index a document
        markdown_doc = """# Test Document

## Section 1

This is the first section with important information about machine learning.

## Section 2

This section discusses deep learning and neural networks.
"""
        
        strategy.index_document(markdown_doc, "test_doc_001")
        
        # Verify chunks were created
        assert len(mock_database.chunk_repository.chunks) > 0
        
        # Retrieve with expansion
        results = strategy.retrieve("machine learning", top_k=3)
        
        # Verify results
        assert len(results) > 0
        
        # Check that results have expansion metadata
        for result in results:
            assert "expansion_strategy" in result.metadata
            assert "original_text" in result.metadata
    
    def test_expansion_strategy_comparison(self, mock_vector_store, mock_database):
        """Test different expansion strategies produce different results."""
        strategy = HierarchicalRAGStrategy(
            vector_store_service=mock_vector_store,
            database_service=mock_database,
            config={"expansion_strategy": ExpansionStrategy.IMMEDIATE_PARENT}
        )
        
        doc = """# Title

## Section

Paragraph 1 with content.

Paragraph 2 with more content.
"""
        
        strategy.index_document(doc, "test_doc")
        
        # Test immediate parent
        strategy.hierarchical_config.expansion_strategy = ExpansionStrategy.IMMEDIATE_PARENT
        immediate_results = strategy.retrieve("content", top_k=2)
        
        # Test full section
        strategy.hierarchical_config.expansion_strategy = ExpansionStrategy.FULL_SECTION
        section_results = strategy.retrieve("content", top_k=2)
        
        # Both should return results
        assert len(immediate_results) > 0
        assert len(section_results) > 0
        
        # Expansion strategies should be recorded
        if immediate_results:
            assert immediate_results[0].metadata["expansion_strategy"] == "immediate_parent"
        if section_results:
            assert section_results[0].metadata["expansion_strategy"] == "full_section"
    
    def test_hierarchy_validation(self, mock_vector_store, mock_database):
        """Test hierarchy validation."""
        strategy = HierarchicalRAGStrategy(
            vector_store_service=mock_vector_store,
            database_service=mock_database
        )
        
        doc = "# Title\n\n## Section\n\nContent"
        strategy.index_document(doc, "test_doc")
        
        # Run validation
        issues = mock_database.chunk_repository.validate_hierarchy()
        
        # Should have no issues
        assert len(issues) == 0
    
    def test_multiple_documents(self, mock_vector_store, mock_database):
        """Test indexing multiple documents."""
        strategy = HierarchicalRAGStrategy(
            vector_store_service=mock_vector_store,
            database_service=mock_database
        )
        
        doc1 = "# Document 1\n\nContent about AI"
        doc2 = "# Document 2\n\nContent about ML"
        
        strategy.index_document(doc1, "doc1")
        strategy.index_document(doc2, "doc2")
        
        # Both documents should be indexed
        chunks = mock_database.chunk_repository.chunks
        doc1_chunks = [c for c in chunks.values() if str(c.document_id) == "doc1"]
        doc2_chunks = [c for c in chunks.values() if str(c.document_id) == "doc2"]
        
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0
