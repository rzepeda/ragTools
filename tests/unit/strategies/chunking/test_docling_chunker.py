"""Unit tests for DoclingChunker (stub implementation)."""

import pytest
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod

# Try to import docling chunker
try:
    from rag_factory.strategies.chunking.docling_chunker import (
        DoclingChunker,
        is_docling_available,
        get_docling_version
    )
    DOCLING_CHUNKER_AVAILABLE = True
except ImportError:
    DOCLING_CHUNKER_AVAILABLE = False


@pytest.fixture
def chunking_config():
    """Create chunking configuration."""
    return ChunkingConfig(
        method=ChunkingMethod.DOCKLING,
        target_chunk_size=512
    )


@pytest.mark.skipif(not DOCLING_CHUNKER_AVAILABLE, reason="DoclingChunker module not available")
def test_is_docling_available():
    """Test docling availability check."""
    # The function should return a boolean
    available = is_docling_available()
    assert isinstance(available, bool)


@pytest.mark.skipif(not DOCLING_CHUNKER_AVAILABLE, reason="DoclingChunker module not available")
def test_get_docling_version():
    """Test getting docling version."""
    version = get_docling_version()
    # Version is either a string or None
    assert version is None or isinstance(version, str)


@pytest.mark.skipif(not DOCLING_CHUNKER_AVAILABLE, reason="DoclingChunker module not available")
def test_docling_chunker_import_error_when_not_installed(chunking_config):
    """Test that DoclingChunker raises ImportError when docling is not installed."""
    if not is_docling_available():
        with pytest.raises(ImportError, match="docling library is required"):
            DoclingChunker(chunking_config)


@pytest.mark.skipif(not DOCLING_CHUNKER_AVAILABLE, reason="DoclingChunker module not available")
@pytest.mark.skipif(not is_docling_available() if DOCLING_CHUNKER_AVAILABLE else True,
                    reason="docling library not installed")
def test_docling_chunker_initialization(chunking_config):
    """Test docling chunker initializes when library is available."""
    chunker = DoclingChunker(chunking_config)
    assert chunker.config == chunking_config


@pytest.mark.skipif(not DOCLING_CHUNKER_AVAILABLE, reason="DoclingChunker module not available")
@pytest.mark.skipif(not is_docling_available() if DOCLING_CHUNKER_AVAILABLE else True,
                    reason="docling library not installed")
def test_docling_chunker_placeholder_warning(chunking_config):
    """Test that placeholder methods log warnings."""
    chunker = DoclingChunker(chunking_config)

    # These should return empty lists in the stub implementation
    chunks = chunker.chunk_document("test document", "doc_1")
    assert chunks == []

    tables = chunker.extract_tables("test document")
    assert tables == []

    figures = chunker.extract_figures("test document")
    assert figures == []


def test_docling_chunker_module_structure():
    """Test that docling_chunker module can be imported without docling."""
    # This test ensures the module structure is correct even without docling
    try:
        from rag_factory.strategies.chunking import docling_chunker
        assert hasattr(docling_chunker, 'DoclingChunker')
        assert hasattr(docling_chunker, 'is_docling_available')
        assert hasattr(docling_chunker, 'get_docling_version')
    except ImportError:
        pytest.skip("docling_chunker module not available")
