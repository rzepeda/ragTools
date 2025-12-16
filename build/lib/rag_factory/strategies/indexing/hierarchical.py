"""Hierarchical indexing strategy implementation.

This module implements the hierarchical indexing strategy that creates
parent-child chunk relationships for context expansion during retrieval.
"""

import logging
import re
from typing import List, Dict, Any, Set, Optional

from rag_factory.core.indexing_interface import (
    IIndexingStrategy,
    IndexingContext,
    IndexingResult
)
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.factory import register_rag_strategy as register_strategy

logger = logging.getLogger(__name__)


@register_strategy("HierarchicalIndexing")
class HierarchicalIndexing(IIndexingStrategy):
    """Creates hierarchical chunk relationships.
    
    This strategy builds a multi-level hierarchy of chunks:
    - Level 0: Full document
    - Level 1: Sections (split by headings)
    - Level 2: Paragraphs
    
    Each chunk stores parent-child relationships and path metadata
    for context expansion during retrieval.
    
    Produces:
        - CHUNKS: Document chunks for retrieval
        - HIERARCHY: Parent-child relationships between chunks
        - DATABASE: Data persisted to database
        
    Requires:
        - DATABASE: Service for storing chunks with hierarchy
    """

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.

        Returns:
            Set containing CHUNKS, HIERARCHY, VECTORS, and DATABASE capabilities
        """
        return {
            IndexCapability.CHUNKS,
            IndexCapability.HIERARCHY,
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Set containing EMBEDDING and DATABASE service dependencies
        """
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }

    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Create hierarchical chunks with parent-child relationships.

        Hierarchy levels:
        - Level 0: Full document
        - Level 1: Sections (e.g., headings)
        - Level 2: Paragraphs

        Args:
            documents: List of documents to index
            context: Indexing context with database service

        Returns:
            IndexingResult with capabilities and metrics
        """
        # Configuration
        max_depth = self.config.get('max_depth', 2)
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            if not text:
                logger.warning("Skipping document with no text: %s", doc.get('id', 'unknown'))
                continue

            doc_id = doc.get('id', 'unknown')

            # Create hierarchy
            hierarchy = self._build_hierarchy(text, max_depth)

            # Flatten and assign IDs with relationships
            chunks = self._flatten_hierarchy(hierarchy, doc_id)
            # Add original document metadata to all chunks
            if 'metadata' in doc:
                for chunk in chunks:
                    chunk['metadata'] = chunk.get('metadata', {})
                    chunk['metadata'].update(doc['metadata'])

            all_chunks.extend(chunks)

        # Generate embeddings for all chunks
        if all_chunks and self.deps.embedding_service:
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = await self.deps.embedding_service.embed_batch(chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk['embedding'] = embedding

        # Store chunks with hierarchy metadata
        if all_chunks:
            await context.database.store_chunks_with_hierarchy(all_chunks)

        # Calculate metrics
        avg_chunks_per_doc = len(all_chunks) / len(documents) if documents else 0

        # Count chunks by level
        level_counts: Dict[int, int] = {}
        for chunk in all_chunks:
            level = chunk.get('level', 0)
            level_counts[level] = level_counts.get(level, 0) + 1

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'max_depth': max_depth,
                'avg_chunks_per_doc': avg_chunks_per_doc,
                'level_counts': level_counts
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )

    def _build_hierarchy(self, text: str, max_depth: int) -> Dict[str, Any]:
        """Build hierarchical structure from text.
        
        Args:
            text: Document text to hierarchically structure
            max_depth: Maximum depth of hierarchy (0, 1, or 2)
            
        Returns:
            Dictionary representing the hierarchy tree
        """
        # Level 0: Full document
        hierarchy = {
            'level': 0,
            'text': text,
            'children': []
        }

        if max_depth >= 1:
            # Level 1: Split by sections (headings)
            sections = self._split_by_headings(text)
            for section in sections:
                section_node = {
                    'level': 1,
                    'text': section,
                    'children': []
                }

                if max_depth >= 2:
                    # Level 2: Split by paragraphs
                    paragraphs = self._split_by_paragraphs(section)
                    for para in paragraphs:
                        para_node = {
                            'level': 2,
                            'text': para,
                            'children': []
                        }
                        section_node['children'].append(para_node)

                hierarchy['children'].append(section_node)

        return hierarchy

    def _flatten_hierarchy(
        self,
        hierarchy: Dict[str, Any],
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """Flatten hierarchy into chunks with parent references.
        
        Args:
            hierarchy: Hierarchical tree structure
            doc_id: Document ID for chunk identification
            
        Returns:
            List of chunk dictionaries with hierarchy metadata
        """
        chunks = []

        def traverse(
            node: Dict[str, Any],
            parent_id: Optional[str] = None,
            path: Optional[List[int]] = None
        ) -> None:
            """Recursively traverse hierarchy and create chunks."""
            if path is None:
                path = []

            # Create chunk ID from document ID and path
            if path:
                chunk_id = f"{doc_id}_{'_'.join(map(str, path))}"
            else:
                chunk_id = f"{doc_id}_root"

            chunk = {
                'id': chunk_id,
                'document_id': doc_id,
                'text': node['text'],
                'level': node['level'],
                'parent_id': parent_id,
                'path': path.copy(),
                'metadata': {
                    'strategy': 'hierarchical'
                }
            }
            chunks.append(chunk)

            # Traverse children
            for i, child in enumerate(node['children']):
                traverse(child, chunk_id, path + [i])

        traverse(hierarchy)
        return chunks

    def _split_by_headings(self, text: str) -> List[str]:
        """Split text by markdown/HTML headings.
        
        Detects markdown headings (# Header) and HTML headings (<h1>Header</h1>).
        If no headings are found, returns the entire text as a single section.

        Args:
            text: Text to split
            
        Returns:
            List of section texts
        """
        # Pattern to match markdown headings (# Header) or HTML headings (<h1>Header</h1>)
        # Markdown: one or more # followed by space and text
        # HTML: <h1> through <h6> tags
        heading_pattern = r'(?:^|\n)(#{1,6}\s+.+?(?=\n|$)|<h[1-6]>.*?</h[1-6]>)'
        # Find all headings
        headings = list(re.finditer(heading_pattern, text, re.MULTILINE | re.DOTALL))
        if not headings:
            # No headings found, return entire text as single section
            return [text.strip()] if text.strip() else []

        sections = []
        for i, match in enumerate(headings):
            # Start of this section is the heading
            start = match.start()

            # End of this section is the start of next heading or end of text
            if i + 1 < len(headings):
                end = headings[i + 1].start()
            else:
                end = len(text)

            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)

        # If we found headings but no sections (edge case), return full text
        if not sections:
            return [text.strip()] if text.strip() else []
        return sections

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks.
        
        Paragraphs are separated by double newlines or more.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraph texts
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
