# Docling Chunker - Implementation Guide

## Overview

The `DoclingChunker` is a stub implementation ready for integration with the `docling` library for advanced PDF and document processing. This guide explains how to complete the implementation when the library is available.

## Installation

```bash
pip install docling
```

## Current Status

✅ **Completed:**
- API interface defined
- Import error handling
- Optional import mechanism in `__init__.py`
- Basic test structure
- Helper functions (`is_docling_available()`, `get_docling_version()`)

⚠️ **Pending:**
- Actual docling library integration
- PDF parsing implementation
- Table extraction
- Figure extraction
- Document layout analysis

## Implementation Tasks

### 1. Initialize Docling Parser

In `__init__()`, initialize the docling document parser:

```python
def __init__(self, config: ChunkingConfig, docling_config: Optional[Dict[str, Any]] = None):
    super().__init__(config)

    if not DOCLING_AVAILABLE:
        raise ImportError("docling library is required")

    # Initialize docling parser
    self.parser = docling.DocumentParser(
        layout_analysis=config.extra_config.get('layout_analysis', True),
        table_extraction=config.extra_config.get('table_extraction', True),
        **self.docling_config
    )
```

### 2. Implement PDF Chunking

Replace the placeholder in `chunk_pdf()`:

```python
def chunk_pdf(self, pdf_path: str, document_id: str) -> List[Chunk]:
    # Parse PDF with docling
    document = self.parser.parse_pdf(pdf_path)

    # Extract structure
    sections = document.sections

    chunks = []
    for i, section in enumerate(sections):
        # Create chunk from section
        chunk_text = section.text

        # Extract metadata
        metadata = ChunkMetadata(
            chunk_id=f"{document_id}_chunk_{i}",
            source_document_id=document_id,
            position=i,
            start_char=section.start_pos,
            end_char=section.end_pos,
            section_hierarchy=section.hierarchy,
            chunking_method=ChunkingMethod.DOCKLING,
            token_count=self._count_tokens(chunk_text),
            metadata={
                'page_number': section.page,
                'bbox': section.bounding_box
            }
        )

        chunks.append(Chunk(text=chunk_text, metadata=metadata))

    return chunks
```

### 3. Implement Table Extraction

Replace the placeholder in `extract_tables()`:

```python
def extract_tables(self, document: str) -> List[Dict[str, Any]]:
    doc = self.parser.parse(document)

    tables = []
    for table in doc.tables:
        tables.append({
            'data': table.to_dataframe(),  # Convert to pandas DataFrame
            'caption': table.caption,
            'position': table.position,
            'cells': table.cells,
            'bbox': table.bounding_box
        })

    return tables
```

### 4. Implement Figure Extraction

Replace the placeholder in `extract_figures()`:

```python
def extract_figures(self, document: str) -> List[Dict[str, Any]]:
    doc = self.parser.parse(document)

    figures = []
    for figure in doc.figures:
        figures.append({
            'image': figure.image_data,
            'caption': figure.caption,
            'position': figure.position,
            'bbox': figure.bounding_box,
            'page': figure.page_number
        })

    return figures
```

### 5. Implement Document Chunking

Replace the placeholder in `chunk_document()`:

```python
def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
    # Detect document type
    if document.endswith('.pdf'):
        return self.chunk_pdf(document, document_id)

    # Parse document
    doc = self.parser.parse(document)

    chunks = []
    for i, element in enumerate(doc.elements):
        # Handle different element types
        if element.type == 'paragraph':
            text = element.text
        elif element.type == 'table':
            # Convert table to markdown or keep as structured data
            text = self._table_to_markdown(element)
        elif element.type == 'list':
            text = self._list_to_text(element)
        else:
            text = element.text

        # Create chunk
        token_count = self._count_tokens(text)

        # Apply size constraints
        if token_count > self.config.max_chunk_size:
            # Split large chunks
            sub_chunks = self._split_large_element(text, document_id, i)
            chunks.extend(sub_chunks)
        else:
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                source_document_id=document_id,
                position=i,
                start_char=element.start_pos,
                end_char=element.end_pos,
                section_hierarchy=element.hierarchy,
                chunking_method=ChunkingMethod.DOCKLING,
                token_count=token_count,
                metadata={
                    'element_type': element.type,
                    'page': element.page
                }
            )

            chunks.append(Chunk(text=text, metadata=metadata))

    return chunks
```

## Testing

Update the tests in `test_docling_chunker.py` to test actual functionality:

```python
@pytest.mark.skipif(not is_docling_available(), reason="docling not installed")
def test_docling_pdf_chunking():
    config = ChunkingConfig(method=ChunkingMethod.DOCKLING)
    chunker = DoclingChunker(config)

    chunks = chunker.chunk_pdf("test.pdf", "test_doc")

    assert len(chunks) > 0
    assert all(c.metadata.chunking_method == ChunkingMethod.DOCKLING for c in chunks)
```

## Configuration Options

Add docling-specific options to `ChunkingConfig.extra_config`:

```python
config = ChunkingConfig(
    method=ChunkingMethod.DOCKLING,
    extra_config={
        'layout_analysis': True,      # Enable layout analysis
        'table_extraction': True,      # Extract tables
        'figure_extraction': True,     # Extract figures
        'ocr_enabled': False,          # Use OCR for scanned PDFs
        'preserve_formatting': True,   # Preserve text formatting
        'reading_order': 'natural'     # 'natural' or 'column'
    }
)
```

## API Reference

### Docling Library Documentation

- Official Docs: [Link to docling documentation when available]
- GitHub: [Link to docling repository]
- Examples: See docling examples for PDF processing patterns

### Expected Docling API

Based on similar libraries, docling likely provides:

```python
# Document parsing
docling.DocumentParser(config)
parser.parse(file_path)
parser.parse_pdf(pdf_path)

# Document structure
document.sections       # List of sections
document.paragraphs     # List of paragraphs
document.tables         # List of tables
document.figures        # List of figures

# Element properties
element.text           # Text content
element.type           # Element type (paragraph, table, etc.)
element.hierarchy      # Section hierarchy
element.page           # Page number
element.bounding_box   # Position on page
```

## Integration Checklist

- [ ] Install docling library
- [ ] Initialize document parser in `__init__()`
- [ ] Implement `chunk_document()` with docling parsing
- [ ] Implement `chunk_pdf()` for PDF-specific processing
- [ ] Implement `extract_tables()` with table detection
- [ ] Implement `extract_figures()` with image extraction
- [ ] Add helper methods for element conversion
- [ ] Update tests to use real docling functionality
- [ ] Add integration tests with sample PDF files
- [ ] Update documentation with docling-specific examples
- [ ] Performance testing with various document types

## Notes

- The stub is designed to raise helpful errors when docling is not available
- All placeholder methods return empty lists or raise warnings
- The import mechanism allows the rest of the chunking module to work without docling
- Tests are structured to skip when docling is unavailable

## Example Usage (Future)

Once implemented, usage will be:

```python
from rag_factory.strategies.chunking import DoclingChunker, ChunkingConfig

config = ChunkingConfig(method=ChunkingMethod.DOCKLING)
chunker = DoclingChunker(config)

# Chunk a PDF
chunks = chunker.chunk_pdf("document.pdf", "doc_1")

# Extract tables
tables = chunker.extract_tables("document.pdf")

# Extract figures
figures = chunker.extract_figures("document.pdf")
```
