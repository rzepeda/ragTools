"""Unit tests for HierarchyBuilder."""

import pytest
from rag_factory.strategies.hierarchical.hierarchy_builder import HierarchyBuilder
from rag_factory.strategies.hierarchical.models import (
    HierarchyLevel,
    HierarchicalConfig
)


class TestHierarchyBuilder:
    """Tests for HierarchyBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create a HierarchyBuilder instance."""
        return HierarchyBuilder()
    
    @pytest.fixture
    def markdown_text(self):
        """Sample markdown text."""
        return """# Document Title

## Section 1

This is the first section with some content.

### Subsection 1.1

More detailed content here.

## Section 2

This is the second section.

### Subsection 2.1

Another subsection with content.
"""
    
    @pytest.fixture
    def plain_text(self):
        """Sample plain text without markdown."""
        return """This is a plain text document with enough content in each paragraph to meet the minimum chunk size requirements for hierarchical chunking.

It has multiple paragraphs that are separated by double newlines, which allows the hierarchy builder to detect the document structure.

Each paragraph contains sufficient text to be considered a valid chunk, ensuring that the minimum token count threshold is met during the chunking process.

This allows for hierarchical chunking based on paragraphs, creating a proper parent-child relationship structure in the document hierarchy.
"""
    
    def test_build_creates_root_chunk(self, builder):
        """Test that build creates a root document chunk."""
        hierarchy = builder.build("Test document", "doc1")
        
        assert hierarchy.root_chunk is not None
        assert hierarchy.root_chunk.hierarchy_level == HierarchyLevel.DOCUMENT
        assert hierarchy.root_chunk.parent_chunk_id is None
        assert hierarchy.root_chunk.document_id == "doc1"
    
    def test_is_markdown_detection(self, builder, markdown_text, plain_text):
        """Test markdown detection."""
        assert builder._is_markdown(markdown_text) is True
        assert builder._is_markdown(plain_text) is False
    
    def test_markdown_hierarchy_structure(self, builder, markdown_text):
        """Test that markdown creates proper hierarchy."""
        hierarchy = builder.build(markdown_text, "doc1")
        
        # Should have root + sections
        assert len(hierarchy.all_chunks) > 1
        
        # Check that we have different hierarchy levels
        assert HierarchyLevel.DOCUMENT in hierarchy.levels
        assert HierarchyLevel.SECTION in hierarchy.levels
    
    def test_split_by_headers(self, builder, markdown_text):
        """Test header splitting."""
        sections = builder._split_by_headers(markdown_text)
        
        # Should have multiple sections
        assert len(sections) > 0
        
        # Check first section
        assert sections[0][0] == 1  # Header level
        assert "Document Title" in sections[0][1]  # Header text
    
    def test_paragraph_hierarchy(self, plain_text):
        """Test paragraph-based hierarchy for plain text."""
        # Use smaller min_chunk_size to ensure paragraphs are created
        config = HierarchicalConfig(min_chunk_size=10)
        builder = HierarchyBuilder(config)
        hierarchy = builder.build(plain_text, "doc1")
        
        # Should have root + paragraphs
        assert len(hierarchy.all_chunks) > 1
        
        # All non-root chunks should be at section or paragraph level
        for chunk_id, chunk in hierarchy.all_chunks.items():
            if chunk.hierarchy_level != HierarchyLevel.DOCUMENT:
                assert chunk.parent_chunk_id is not None
    
    def test_hierarchy_metadata(self, builder, markdown_text):
        """Test that hierarchy metadata is populated."""
        hierarchy = builder.build(markdown_text, "doc1")
        
        for chunk_id, chunk in hierarchy.all_chunks.items():
            # All chunks should have metadata
            assert chunk.hierarchy_metadata is not None
            assert chunk.hierarchy_metadata.position_in_parent >= 0
            assert chunk.hierarchy_metadata.total_siblings >= 0
            assert chunk.hierarchy_metadata.depth_from_root >= 0
    
    def test_parent_child_relationships(self, builder, markdown_text):
        """Test that parent-child relationships are correct."""
        hierarchy = builder.build(markdown_text, "doc1")
        
        # Find a non-root chunk
        child_chunks = [
            c for c in hierarchy.all_chunks.values()
            if c.parent_chunk_id is not None
        ]
        
        assert len(child_chunks) > 0
        
        # Verify parent exists for each child
        for child in child_chunks:
            assert child.parent_chunk_id in hierarchy.all_chunks
    
    def test_min_chunk_size_respected(self, builder):
        """Test that very small chunks are filtered out."""
        config = HierarchicalConfig(min_chunk_size=50)
        builder = HierarchyBuilder(config)
        
        text = "# Title\n\nShort.\n\nThis is a longer paragraph with enough content to meet the minimum size requirement."
        hierarchy = builder.build(text, "doc1")
        
        # Very short paragraphs should be filtered
        for chunk in hierarchy.all_chunks.values():
            if chunk.hierarchy_level == HierarchyLevel.PARAGRAPH:
                assert len(chunk.text.split()) >= config.min_chunk_size or chunk.hierarchy_level == HierarchyLevel.DOCUMENT
    
    def test_max_hierarchy_depth(self, builder):
        """Test that hierarchy depth is limited."""
        config = HierarchicalConfig(max_hierarchy_depth=2)
        builder = HierarchyBuilder(config)
        
        text = "# L1\n\n## L2\n\n### L3\n\n#### L4\n\nContent"
        hierarchy = builder.build(text, "doc1")
        
        # No chunk should exceed max depth
        for chunk in hierarchy.all_chunks.values():
            assert chunk.hierarchy_level.value < config.max_hierarchy_depth
    
    def test_empty_document(self, builder):
        """Test handling of empty document."""
        hierarchy = builder.build("", "doc1")
        
        # Should still have root chunk
        assert hierarchy.root_chunk is not None
        assert len(hierarchy.all_chunks) == 1
    
    def test_document_with_no_structure(self, builder):
        """Test document with no clear structure."""
        text = "Just a single line of text without any structure."
        hierarchy = builder.build(text, "doc1")
        
        # Should have at least root
        assert len(hierarchy.all_chunks) >= 1
        assert hierarchy.root_chunk.text == text


class TestHierarchyBuilderEdgeCases:
    """Test edge cases for HierarchyBuilder."""
    
    def test_nested_headers_same_level(self):
        """Test multiple headers at the same level."""
        builder = HierarchyBuilder()
        text = """# Header 1

Content 1

# Header 2

Content 2

# Header 3

Content 3
"""
        hierarchy = builder.build(text, "doc1")
        
        # Should have multiple section-level chunks
        section_chunks = [
            c for c in hierarchy.all_chunks.values()
            if c.hierarchy_level == HierarchyLevel.SECTION
        ]
        assert len(section_chunks) >= 2
    
    def test_header_without_content(self):
        """Test headers with no content."""
        builder = HierarchyBuilder()
        text = """# Header 1

## Subheader 1

## Subheader 2

# Header 2
"""
        hierarchy = builder.build(text, "doc1")
        
        # Should handle gracefully
        assert hierarchy.root_chunk is not None
    
    def test_very_long_document(self):
        """Test with a very long document."""
        builder = HierarchyBuilder()
        
        # Create a long document
        sections = []
        for i in range(50):
            sections.append(f"## Section {i}\n\n" + "Content paragraph. " * 100)
        
        text = "# Large Document\n\n" + "\n\n".join(sections)
        hierarchy = builder.build(text, "doc1")
        
        # Should handle large documents
        assert len(hierarchy.all_chunks) > 50
