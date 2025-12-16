"""
Hierarchy Builder for Document Structure Detection.

This module builds hierarchical chunk structures from documents,
detecting sections, paragraphs, and creating parent-child relationships.
"""

import re
import uuid
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .models import (
    HierarchicalChunk,
    ChunkHierarchy,
    HierarchyLevel,
    HierarchyMetadata,
    HierarchicalConfig
)


class HierarchyBuilder:
    """Builds hierarchical chunk structures from documents.
    
    Detects document structure (headers, sections, paragraphs) and creates
    parent-child relationships between chunks.
    """
    
    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """Initialize the hierarchy builder.
        
        Args:
            config: Configuration for hierarchy building
        """
        self.config = config or HierarchicalConfig()
        self._token_estimator = lambda text: len(text.split())  # Simple token estimation
    
    def build(self, text: str, document_id: str) -> ChunkHierarchy:
        """Build a hierarchical structure from document text.
        
        Args:
            text: The document text
            document_id: Unique identifier for the document
            
        Returns:
            ChunkHierarchy: The complete hierarchy structure
        """
        # Create root document chunk
        root_id = str(uuid.uuid4())
        root_chunk = HierarchicalChunk(
            chunk_id=root_id,
            document_id=document_id,
            text=text[:self.config.large_chunk_size] if len(text) > self.config.large_chunk_size else text,
            hierarchy_level=HierarchyLevel.DOCUMENT,
            hierarchy_metadata=HierarchyMetadata(
                position_in_parent=0,
                total_siblings=0,
                depth_from_root=0
            ),
            parent_chunk_id=None,
            token_count=self._token_estimator(text)
        )
        
        all_chunks: Dict[str, HierarchicalChunk] = {root_id: root_chunk}
        levels: Dict[HierarchyLevel, List[str]] = defaultdict(list)
        levels[HierarchyLevel.DOCUMENT].append(root_id)
        
        # Detect if this is markdown with headers
        if self._is_markdown(text):
            self._build_markdown_hierarchy(text, document_id, root_id, all_chunks, levels)
        else:
            # Fall back to paragraph-based chunking
            self._build_paragraph_hierarchy(text, document_id, root_id, all_chunks, levels)
        
        return ChunkHierarchy(
            document_id=document_id,
            root_chunk=root_chunk,
            all_chunks=all_chunks,
            levels=dict(levels)
        )
    
    def _is_markdown(self, text: str) -> bool:
        """Check if text appears to be markdown with headers.
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if text contains markdown headers
        """
        # Look for markdown headers (# Header)
        header_pattern = r'^#{1,6}\s+.+$'
        return bool(re.search(header_pattern, text, re.MULTILINE))
    
    def _build_markdown_hierarchy(
        self,
        text: str,
        document_id: str,
        root_id: str,
        all_chunks: Dict[str, HierarchicalChunk],
        levels: Dict[HierarchyLevel, List[str]]
    ) -> None:
        """Build hierarchy from markdown structure.
        
        Args:
            text: The markdown text
            document_id: Document identifier
            root_id: Root chunk ID
            all_chunks: Dictionary to populate with chunks
            levels: Dictionary to populate with level mappings
        """
        # Split by headers
        sections = self._split_by_headers(text)
        
        for i, (header_level, header_text, content) in enumerate(sections):
            if not content.strip() and not header_text:
                continue
            
            # Map markdown header level to hierarchy level
            # # Header (level 1) -> SECTION (1)
            # ## Header (level 2) -> SECTION (1) or PARAGRAPH (2) depending on nesting
            # For simplicity, treat all headers as SECTION level
            hierarchy_level = HierarchyLevel.SECTION
            
            # Create section chunk
            section_id = str(uuid.uuid4())
            section_text = f"{header_text}\n\n{content}" if header_text else content
            
            section_chunk = HierarchicalChunk(
                chunk_id=section_id,
                document_id=document_id,
                text=section_text,
                hierarchy_level=hierarchy_level,
                hierarchy_metadata=HierarchyMetadata(
                    position_in_parent=i,
                    total_siblings=len(sections),
                    depth_from_root=1  # Sections are always depth 1 from root
                ),
                parent_chunk_id=root_id,
                token_count=self._token_estimator(section_text),
                metadata={"header": header_text, "header_level": header_level}
            )
            
            all_chunks[section_id] = section_chunk
            levels[hierarchy_level].append(section_id)
            
            # Split section into paragraphs if it's large enough
            if self._token_estimator(content) > self.config.small_chunk_size:
                self._build_paragraph_chunks(
                    content,
                    document_id,
                    section_id,
                    2,  # Paragraphs are depth 2
                    all_chunks,
                    levels
                )
    
    def _split_by_headers(self, text: str) -> List[Tuple[int, str, str]]:
        """Split markdown text by headers.
        
        Args:
            text: The markdown text
            
        Returns:
            List of (header_level, header_text, content) tuples
        """
        sections = []
        lines = text.split('\n')
        current_header_level = 0
        current_header = ""
        current_content = []
        
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # Save previous section
                if current_content or current_header:
                    sections.append((
                        current_header_level,
                        current_header,
                        '\n'.join(current_content)
                    ))
                
                # Start new section
                current_header_level = len(match.group(1))
                current_header = match.group(2)
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content or current_header:
            sections.append((
                current_header_level,
                current_header,
                '\n'.join(current_content)
            ))
        
        return sections
    
    def _build_paragraph_hierarchy(
        self,
        text: str,
        document_id: str,
        root_id: str,
        all_chunks: Dict[str, HierarchicalChunk],
        levels: Dict[HierarchyLevel, List[str]]
    ) -> None:
        """Build hierarchy from paragraph structure.
        
        Args:
            text: The text to chunk
            document_id: Document identifier
            root_id: Root chunk ID
            all_chunks: Dictionary to populate with chunks
            levels: Dictionary to populate with level mappings
        """
        self._build_paragraph_chunks(
            text,
            document_id,
            root_id,
            1,  # Start at section level
            all_chunks,
            levels
        )
    
    def _build_paragraph_chunks(
        self,
        text: str,
        document_id: str,
        parent_id: str,
        depth: int,
        all_chunks: Dict[str, HierarchicalChunk],
        levels: Dict[HierarchyLevel, List[str]]
    ) -> None:
        """Build paragraph-level chunks.
        
        Args:
            text: The text to chunk
            document_id: Document identifier
            parent_id: Parent chunk ID
            depth: Current depth in hierarchy
            all_chunks: Dictionary to populate with chunks
            levels: Dictionary to populate with level mappings
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Determine hierarchy level
        hierarchy_level = HierarchyLevel(min(depth, self.config.max_hierarchy_depth - 1))
        
        for i, paragraph in enumerate(paragraphs):
            # Skip very small paragraphs
            if self._token_estimator(paragraph) < self.config.min_chunk_size:
                continue
            
            # Create paragraph chunk
            para_id = str(uuid.uuid4())
            para_chunk = HierarchicalChunk(
                chunk_id=para_id,
                document_id=document_id,
                text=paragraph,
                hierarchy_level=hierarchy_level,
                hierarchy_metadata=HierarchyMetadata(
                    position_in_parent=i,
                    total_siblings=len(paragraphs),
                    depth_from_root=depth
                ),
                parent_chunk_id=parent_id,
                token_count=self._token_estimator(paragraph)
            )
            
            all_chunks[para_id] = para_chunk
            levels[hierarchy_level].append(para_id)
            
            # If paragraph is large, split into sentences
            if (self._token_estimator(paragraph) > self.config.large_chunk_size and 
                depth < self.config.max_hierarchy_depth - 1):
                self._build_sentence_chunks(
                    paragraph,
                    document_id,
                    para_id,
                    depth + 1,
                    all_chunks,
                    levels
                )
    
    def _build_sentence_chunks(
        self,
        text: str,
        document_id: str,
        parent_id: str,
        depth: int,
        all_chunks: Dict[str, HierarchicalChunk],
        levels: Dict[HierarchyLevel, List[str]]
    ) -> None:
        """Build sentence-level chunks.
        
        Args:
            text: The text to chunk
            document_id: Document identifier
            parent_id: Parent chunk ID
            depth: Current depth in hierarchy
            all_chunks: Dictionary to populate with chunks
            levels: Dictionary to populate with level mappings
        """
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'[.!?]+\s+', text)
        
        hierarchy_level = HierarchyLevel.SENTENCE
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or self._token_estimator(sentence) < self.config.min_chunk_size:
                continue
            
            sent_id = str(uuid.uuid4())
            sent_chunk = HierarchicalChunk(
                chunk_id=sent_id,
                document_id=document_id,
                text=sentence,
                hierarchy_level=hierarchy_level,
                hierarchy_metadata=HierarchyMetadata(
                    position_in_parent=i,
                    total_siblings=len(sentences),
                    depth_from_root=depth
                ),
                parent_chunk_id=parent_id,
                token_count=self._token_estimator(sentence)
            )
            
            all_chunks[sent_id] = sent_chunk
            levels[hierarchy_level].append(sent_id)
