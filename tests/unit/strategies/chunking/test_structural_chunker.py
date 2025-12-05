"""Unit tests for StructuralChunker."""

import pytest
from rag_factory.strategies.chunking.structural_chunker import StructuralChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod


@pytest.fixture
def chunking_config():
    """Create chunking configuration."""
    return ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512,
        respect_headers=True,
        respect_paragraphs=True
    )


def test_structural_chunker_initialization(chunking_config):
    """Test structural chunker initializes correctly."""
    chunker = StructuralChunker(chunking_config)
    assert chunker.config == chunking_config


def test_is_markdown_detection(chunking_config):
    """Test markdown detection."""
    chunker = StructuralChunker(chunking_config)

    markdown_text = "# Header\n\nSome content"
    assert chunker._is_markdown(markdown_text) is True

    plain_text = "Just plain text without headers"
    assert chunker._is_markdown(plain_text) is False


def test_chunk_markdown_with_headers(chunking_config):
    """Test chunking markdown document with headers."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Header 1

Content under header 1.

## Header 1.1

Content under header 1.1.

## Header 1.2

Content under header 1.2.

# Header 2

Content under header 2.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL for chunk in chunks)

    # Check that hierarchy is preserved
    assert any(len(chunk.metadata.section_hierarchy) > 0 for chunk in chunks)


def test_split_by_headers_hierarchy(chunking_config):
    """Test header hierarchy preservation."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Level 1

Content 1

## Level 2

Content 2

### Level 3

Content 3

## Another Level 2

More content
"""

    sections = chunker._split_by_headers(markdown)

    # Verify hierarchy is tracked
    assert len(sections) > 0

    # Find the Level 3 section
    level_3_sections = [s for s in sections if s["level"] == 3]
    if level_3_sections:
        assert "Level 1" in level_3_sections[0]["hierarchy"]
        assert "Level 2" in level_3_sections[0]["hierarchy"]
        assert "Level 3" in level_3_sections[0]["hierarchy"]


def test_chunk_plain_text_by_paragraphs(chunking_config):
    """Test chunking plain text by paragraphs."""
    chunker = StructuralChunker(chunking_config)

    text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content.

Fourth paragraph."""

    chunks = chunker.chunk_document(text, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL for chunk in chunks)


def test_large_section_splitting(chunking_config):
    """Test splitting large sections that exceed max size."""
    # Configure small chunk size for testing
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        min_chunk_size=20,
        target_chunk_size=50,  # Small size to force splitting
        max_chunk_size=150
    )
    chunker = StructuralChunker(config)

    # Create a large section
    large_section = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])

    chunks = chunker._split_large_section(
        large_section,
        "doc_1",
        0,
        ["Test Section"]
    )

    # Should split into multiple chunks
    assert len(chunks) > 1

    # All chunks should have same hierarchy
    assert all(chunk.metadata.section_hierarchy == ["Test Section"] for chunk in chunks)


def test_empty_markdown_document(chunking_config):
    """Test empty markdown document."""
    chunker = StructuralChunker(chunking_config)

    chunks = chunker.chunk_document("", "doc_1")
    assert chunks == []

    chunks = chunker.chunk_document("   ", "doc_1")
    assert chunks == []


def test_markdown_with_code_blocks(chunking_config):
    """Test markdown with code blocks."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Code Example

Here is some code:

```python
def hello():
    print("Hello, world!")
```

More content after code.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    # Code block should be preserved in chunk
    assert any("```python" in chunk.text for chunk in chunks)


def test_multiple_documents_batch(chunking_config):
    """Test chunking multiple documents in batch."""
    chunker = StructuralChunker(chunking_config)

    documents = [
        {"text": "# Doc 1\n\nContent 1", "id": "doc_1"},
        {"text": "# Doc 2\n\nContent 2", "id": "doc_2"},
        {"text": "Plain text document", "id": "doc_3"}
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 3
    assert all(isinstance(chunks, list) for chunks in results)

    # Verify document IDs are preserved
    assert results[0][0].metadata.source_document_id == "doc_1"
    assert results[1][0].metadata.source_document_id == "doc_2"
    assert results[2][0].metadata.source_document_id == "doc_3"


def test_split_into_paragraphs(chunking_config):
    """Test paragraph splitting."""
    chunker = StructuralChunker(chunking_config)

    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    paragraphs = chunker._split_into_paragraphs(text)

    assert len(paragraphs) == 3
    assert "First paragraph." in paragraphs[0]
    assert "Second paragraph." in paragraphs[1]
    assert "Third paragraph." in paragraphs[2]


def test_is_atomic_content_code_block(chunking_config):
    """Test atomic content detection for code blocks."""
    chunker = StructuralChunker(chunking_config)

    code_block = """```python
def test():
    return True
```"""

    assert chunker._is_atomic_content(code_block) is True


def test_is_atomic_content_table(chunking_config):
    """Test atomic content detection for tables."""
    chunker = StructuralChunker(chunking_config)

    table = """| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |"""

    assert chunker._is_atomic_content(table) is True


def test_atomic_content_not_split(chunking_config):
    """Test that atomic content is kept intact even if oversized."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        min_chunk_size=5,
        target_chunk_size=10,  # Very small
        max_chunk_size=50,
        keep_code_blocks_intact=True
    )
    chunker = StructuralChunker(config)

    # Large code block
    code = """```python
def long_function():
    return "This is a long code block that exceeds the max chunk size"
```"""

    chunks = chunker._split_large_section(code, "doc_1", 0, [])

    # Should keep as single chunk despite size
    assert len(chunks) == 1


def test_respect_headers_disabled():
    """Test chunking with headers disabled."""
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        respect_headers=False
    )
    chunker = StructuralChunker(config)

    markdown = """# Header 1
Content 1

# Header 2
Content 2"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    # Headers should be treated as regular text
    assert len(chunks) > 0


def test_count_tokens(chunking_config):
    """Test token counting."""
    chunker = StructuralChunker(chunking_config)

    text = "This is a test sentence with several words."
    token_count = chunker._count_tokens(text)

    assert isinstance(token_count, int)
    assert token_count > 0
