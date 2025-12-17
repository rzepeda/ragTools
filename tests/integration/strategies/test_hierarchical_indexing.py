"""Unit tests for HierarchicalIndexing strategy."""

import pytest
from unittest.mock import Mock, AsyncMock

from rag_factory.strategies.indexing.hierarchical import HierarchicalIndexing
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies


@pytest.fixture
def mock_database_service():
    """Create mock database service."""
    service = Mock()
    service.store_chunks_with_hierarchy = AsyncMock()
    return service


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock()
    service.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    service.get_dimension = Mock(return_value=384)
    return service


@pytest.fixture
def indexing_context(mock_database_service):
    """Create indexing context with mock database."""
    return IndexingContext(
        database_service=mock_database_service,
        config={}
    )


@pytest.fixture
def hierarchical_strategy(mock_database_service, mock_embedding_service):
    """Create HierarchicalIndexing strategy instance."""
    config = {'max_depth': 2}
    deps = StrategyDependencies(
        database_service=mock_database_service,
        embedding_service=mock_embedding_service
    )
    return HierarchicalIndexing(config, deps)


class TestHierarchicalIndexing:
    """Test suite for HierarchicalIndexing strategy."""

    def test_produces_capabilities(self, hierarchical_strategy):
        """Test that strategy declares correct capabilities."""
        capabilities = hierarchical_strategy.produces()
        
        assert IndexCapability.CHUNKS in capabilities
        assert IndexCapability.HIERARCHY in capabilities
        assert IndexCapability.VECTORS in capabilities
        assert IndexCapability.DATABASE in capabilities
        assert len(capabilities) == 4

    def test_requires_services(self, hierarchical_strategy):
        """Test that strategy declares correct service dependencies."""
        services = hierarchical_strategy.requires_services()
        
        assert ServiceDependency.DATABASE in services
        assert ServiceDependency.EMBEDDING in services
        assert len(services) == 2

    @pytest.mark.asyncio
    async def test_process_basic_document(self, hierarchical_strategy, indexing_context):
        """Test processing a basic document without headings."""
        documents = [
            {
                'id': 'doc1',
                'text': 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
            }
        ]

        result = await hierarchical_strategy.process(documents, indexing_context)

        # Verify result
        assert result.document_count == 1
        assert result.chunk_count > 0
        assert IndexCapability.CHUNKS in result.capabilities
        assert IndexCapability.HIERARCHY in result.capabilities

        # Verify database was called
        indexing_context.database.store_chunks_with_hierarchy.assert_called_once()
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]
        
        # Should have root + sections + paragraphs
        assert len(stored_chunks) > 0
        
        # Check root chunk
        root_chunk = next(c for c in stored_chunks if c['level'] == 0)
        assert root_chunk['parent_id'] is None
        assert root_chunk['path'] == []

    @pytest.mark.asyncio
    async def test_process_document_with_headings(self, hierarchical_strategy, indexing_context):
        """Test processing a document with markdown headings."""
        documents = [
            {
                'id': 'doc2',
                'text': '''# Introduction
This is the introduction.

# Methods
This is the methods section.

## Subsection
This is a subsection.'''
            }
        ]

        result = await hierarchical_strategy.process(documents, indexing_context)

        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]
        
        # Should have root + sections
        assert any(c['level'] == 0 for c in stored_chunks)  # Root
        assert any(c['level'] == 1 for c in stored_chunks)  # Sections
        
        # Check parent-child relationships
        section_chunks = [c for c in stored_chunks if c['level'] == 1]
        for section in section_chunks:
            assert section['parent_id'] is not None
            assert len(section['path']) > 0

    @pytest.mark.asyncio
    async def test_hierarchy_metadata(self, hierarchical_strategy, indexing_context):
        """Test that hierarchy metadata is correctly set."""
        documents = [
            {
                'id': 'doc3',
                'text': 'Paragraph one.\n\nParagraph two.'
            }
        ]

        await hierarchical_strategy.process(documents, indexing_context)
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]

        for chunk in stored_chunks:
            # All chunks should have required metadata
            assert 'id' in chunk
            assert 'document_id' in chunk
            assert 'text' in chunk
            assert 'level' in chunk
            assert 'path' in chunk
            assert chunk['document_id'] == 'doc3'
            
            # Parent ID should be None only for root
            if chunk['level'] == 0:
                assert chunk['parent_id'] is None
            else:
                assert chunk['parent_id'] is not None

    @pytest.mark.asyncio
    async def test_max_depth_configuration(self, mock_database_service, mock_embedding_service, indexing_context):
        """Test that max_depth configuration is respected."""
        # Create strategy with max_depth = 1
        config = {'max_depth': 1}
        deps = StrategyDependencies(
            database_service=mock_database_service,
            embedding_service=mock_embedding_service
        )
        strategy = HierarchicalIndexing(config, deps)

        documents = [
            {
                'id': 'doc4',
                'text': '# Section\n\nParagraph one.\n\nParagraph two.'
            }
        ]

        result = await strategy.process(documents, indexing_context)
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]

        # With max_depth=1, should only have levels 0 and 1
        levels = {c['level'] for c in stored_chunks}
        assert 0 in levels  # Root
        assert 1 in levels  # Sections
        assert 2 not in levels  # No paragraphs

    @pytest.mark.asyncio
    async def test_empty_document(self, hierarchical_strategy, indexing_context):
        """Test handling of empty documents."""
        documents = [
            {
                'id': 'doc5',
                'text': ''
            }
        ]

        result = await hierarchical_strategy.process(documents, indexing_context)

        # Should handle gracefully
        assert result.document_count == 1
        # Database might not be called if no chunks created
        if indexing_context.database.store_chunks_with_hierarchy.called:
            stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]
            assert len(stored_chunks) == 0

    @pytest.mark.asyncio
    async def test_multiple_documents(self, hierarchical_strategy, indexing_context):
        """Test processing multiple documents."""
        documents = [
            {
                'id': 'doc6',
                'text': 'Document one content.'
            },
            {
                'id': 'doc7',
                'text': 'Document two content.'
            }
        ]

        result = await hierarchical_strategy.process(documents, indexing_context)

        assert result.document_count == 2
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]
        
        # Should have chunks from both documents
        doc_ids = {c['document_id'] for c in stored_chunks}
        assert 'doc6' in doc_ids
        assert 'doc7' in doc_ids

    @pytest.mark.asyncio
    async def test_path_tracking(self, hierarchical_strategy, indexing_context):
        """Test that path metadata correctly tracks hierarchy."""
        documents = [
            {
                'id': 'doc8',
                'text': '# Section 1\n\nPara 1.\n\nPara 2.\n\n# Section 2\n\nPara 3.'
            }
        ]

        await hierarchical_strategy.process(documents, indexing_context)
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]

        # Root should have empty path
        root = next(c for c in stored_chunks if c['level'] == 0)
        assert root['path'] == []

        # Level 1 chunks should have single-element paths
        level1_chunks = [c for c in stored_chunks if c['level'] == 1]
        for chunk in level1_chunks:
            assert len(chunk['path']) == 1
            assert isinstance(chunk['path'][0], int)

        # Level 2 chunks should have two-element paths
        level2_chunks = [c for c in stored_chunks if c['level'] == 2]
        for chunk in level2_chunks:
            assert len(chunk['path']) == 2
            assert all(isinstance(p, int) for p in chunk['path'])

    def test_split_by_headings_markdown(self, hierarchical_strategy):
        """Test splitting text by markdown headings."""
        text = '''# Heading 1
Content 1

## Heading 2
Content 2

# Heading 3
Content 3'''

        sections = hierarchical_strategy._split_by_headings(text)
        
        assert len(sections) >= 2  # At least main sections
        assert any('Heading 1' in s for s in sections)
        assert any('Heading 3' in s for s in sections)

    def test_split_by_headings_no_headings(self, hierarchical_strategy):
        """Test splitting text with no headings."""
        text = 'Just plain text without any headings.'

        sections = hierarchical_strategy._split_by_headings(text)
        
        # Should return entire text as single section
        assert len(sections) == 1
        assert sections[0] == text

    def test_split_by_paragraphs(self, hierarchical_strategy):
        """Test splitting text by paragraphs."""
        text = 'Paragraph one.\n\nParagraph two.\n\nParagraph three.'

        paragraphs = hierarchical_strategy._split_by_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert 'Paragraph one.' in paragraphs[0]
        assert 'Paragraph two.' in paragraphs[1]
        assert 'Paragraph three.' in paragraphs[2]

    def test_split_by_paragraphs_single(self, hierarchical_strategy):
        """Test splitting text with single paragraph."""
        text = 'Just one paragraph.'

        paragraphs = hierarchical_strategy._split_by_paragraphs(text)
        
        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    @pytest.mark.asyncio
    async def test_document_metadata_preserved(self, hierarchical_strategy, indexing_context):
        """Test that original document metadata is preserved in chunks."""
        documents = [
            {
                'id': 'doc9',
                'text': 'Some content.',
                'metadata': {
                    'source': 'test.pdf',
                    'author': 'Test Author'
                }
            }
        ]

        await hierarchical_strategy.process(documents, indexing_context)
        stored_chunks = indexing_context.database.store_chunks_with_hierarchy.call_args[0][0]

        # All chunks should have the original metadata
        for chunk in stored_chunks:
            assert 'metadata' in chunk
            assert chunk['metadata']['source'] == 'test.pdf'
            assert chunk['metadata']['author'] == 'Test Author'
            assert chunk['metadata']['strategy'] == 'hierarchical'
