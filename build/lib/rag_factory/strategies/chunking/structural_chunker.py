"""Structural chunking strategy based on document structure."""

from typing import List, Dict, Any, Optional
import re
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod

logger = logging.getLogger(__name__)


class StructuralChunker(IChunker):
    """Chunks documents based on document structure.

    Preserves headers, paragraphs, code blocks, and tables.
    This approach is fast and maintains the natural organization of documents.

    Attributes:
        config: Chunking configuration
        tokenizer: Tokenizer for counting tokens
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize structural chunker.

        Args:
            config: Chunking configuration

        Raises:
            ImportError: If tiktoken not installed
        """
        super().__init__(config)

        if tiktoken is None:
            logger.warning(
                "tiktoken not installed. Using word count approximation. "
                "Install with: pip install tiktoken"
            )
            self.tokenizer = None
        else:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback.")
                self.tokenizer = None

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document based on structural elements.

        Args:
            document: Document text to chunk
            document_id: Unique document identifier

        Returns:
            List of Chunk objects
        """
        if not document or not document.strip():
            return []

        # Detect document type and chunk accordingly
        if self._is_markdown(document):
            return self._chunk_markdown(document, document_id)
        else:
            return self._chunk_plain_text(document, document_id)

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents.

        Args:
            documents: List of dicts with 'text' and 'id' keys

        Returns:
            List of chunk lists, one per document
        """
        return [
            self.chunk_document(doc["text"], doc["id"])
            for doc in documents
        ]

    def _is_markdown(self, text: str) -> bool:
        """Check if document appears to be markdown.

        Args:
            text: Document text

        Returns:
            True if markdown detected, False otherwise
        """
        # Simple heuristic: look for markdown headers
        return bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE))

    def _chunk_markdown(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk markdown document preserving structure.

        Args:
            document: Markdown document text
            document_id: Document identifier

        Returns:
            List of chunks
        """
        chunks = []

        # Split by headers
        sections = self._split_by_headers(document)

        for i, section in enumerate(sections):
            header = section.get("header", "")
            content = section.get("content", "")
            hierarchy = section.get("hierarchy", [])

            # Combine header with content
            full_text = f"{header}\n{content}".strip() if header else content.strip()

            if not full_text:
                continue

            token_count = self._count_tokens(full_text)

            # If section is too large, split further
            if token_count > self.config.max_chunk_size:
                sub_chunks = self._split_large_section(
                    full_text, document_id, i, hierarchy
                )
                chunks.extend(sub_chunks)
            else:
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{i}",
                    source_document_id=document_id,
                    position=i,
                    start_char=0,
                    end_char=len(full_text),
                    section_hierarchy=hierarchy,
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=token_count
                )
                chunks.append(Chunk(text=full_text, metadata=metadata))

        return chunks

    def _split_by_headers(self, document: str) -> List[Dict[str, Any]]:
        """Split markdown document by headers, preserving hierarchy.

        Args:
            document: Markdown document text

        Returns:
            List of section dictionaries
        """
        lines = document.split("\n")
        sections = []
        current_section = {
            "content": [],
            "header": "",
            "level": 0,
            "hierarchy": []
        }
        hierarchy_stack = []

        for line in lines:
            # Check for markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match and self.config.respect_headers:
                # Save previous section if it has content
                if current_section["content"] or current_section["header"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Update hierarchy stack
                while hierarchy_stack and hierarchy_stack[-1]["level"] >= level:
                    hierarchy_stack.pop()

                hierarchy_stack.append({"level": level, "text": header_text})
                hierarchy = [h["text"] for h in hierarchy_stack]

                current_section = {
                    "content": [],
                    "header": line,
                    "level": level,
                    "hierarchy": hierarchy
                }
            else:
                # Add line to current section
                current_section["content"].append(line)

        # Add final section
        if current_section["content"] or current_section["header"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)

        return sections

    def _split_large_section(
        self,
        text: str,
        document_id: str,
        section_idx: int,
        hierarchy: List[str]
    ) -> List[Chunk]:
        """Split a large section into smaller chunks by paragraphs.

        Args:
            text: Section text
            document_id: Document identifier
            section_idx: Section index
            hierarchy: Section hierarchy

        Returns:
            List of chunks
        """
        # First check if this is atomic content
        if self._is_atomic_content(text):
            # Keep as single chunk even if oversized
            token_count = self._count_tokens(text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{section_idx}",
                source_document_id=document_id,
                position=section_idx,
                start_char=0,
                end_char=len(text),
                section_hierarchy=hierarchy,
                chunking_method=ChunkingMethod.STRUCTURAL,
                token_count=token_count
            )
            return [Chunk(text=text, metadata=metadata)]

        # Split by paragraphs
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_text = []
        current_tokens = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self._count_tokens(para)

            # Check if adding this paragraph would exceed target size
            if (current_tokens + para_tokens > self.config.target_chunk_size
                and current_text):
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{section_idx}_{len(chunks)}",
                    source_document_id=document_id,
                    position=section_idx,
                    start_char=0,
                    end_char=len(chunk_text),
                    section_hierarchy=hierarchy,
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=current_tokens
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                current_text = [para]
                current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Add remaining paragraphs
        if current_text:
            chunk_text = "\n\n".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{section_idx}_{len(chunks)}",
                source_document_id=document_id,
                position=section_idx,
                start_char=0,
                end_char=len(chunk_text),
                section_hierarchy=hierarchy,
                chunking_method=ChunkingMethod.STRUCTURAL,
                token_count=current_tokens
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks if chunks else []

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        if self.config.respect_paragraphs:
            # Split by double newlines (paragraphs)
            paragraphs = re.split(r'\n\n+', text)
            return [p.strip() for p in paragraphs if p.strip()]
        else:
            # Return as single paragraph
            return [text] if text.strip() else []

    def _chunk_plain_text(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk plain text document by paragraphs.

        Args:
            document: Plain text document
            document_id: Document identifier

        Returns:
            List of chunks
        """
        paragraphs = self._split_into_paragraphs(document)

        chunks = []
        current_text = []
        current_tokens = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self._count_tokens(para)

            if (current_tokens + para_tokens > self.config.target_chunk_size
                and current_text):
                # Create chunk
                chunk_text = "\n\n".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    source_document_id=document_id,
                    position=len(chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    section_hierarchy=[],
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=current_tokens
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                current_text = [para]
                current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Add remaining text
        if current_text:
            chunk_text = "\n\n".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                source_document_id=document_id,
                position=len(chunks),
                start_char=0,
                end_char=len(chunk_text),
                section_hierarchy=[],
                chunking_method=ChunkingMethod.STRUCTURAL,
                token_count=current_tokens
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to simple word count approximation (avg 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)

    def _is_atomic_content(self, text: str) -> bool:
        """Check if content should be kept as atomic unit.

        Extends base class method with additional checks.

        Args:
            text: Text to check

        Returns:
            True if atomic, False otherwise
        """
        # Use base class check first
        if super()._is_atomic_content(text):
            return True

        text_stripped = text.strip()

        # Check for code blocks with language specifier
        if text_stripped.startswith("```") and self.config.keep_code_blocks_intact:
            return True

        # Check for indented code blocks (4 spaces)
        lines = text.split("\n")
        if len(lines) > 2:
            indented_lines = sum(1 for line in lines if line.startswith("    "))
            if indented_lines / len(lines) > 0.5 and self.config.keep_code_blocks_intact:
                return True

        # Check for tables with markdown syntax
        if self.config.keep_tables_intact:
            # Look for table rows (lines with multiple |)
            table_lines = [
                line for line in lines
                if "|" in line and line.count("|") >= 2
            ]
            if len(table_lines) >= 2:
                return True

        return False
