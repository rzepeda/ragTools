"""
Parent Retriever for Context Expansion.

This module expands retrieved chunks with parent context using various
expansion strategies (immediate parent, full section, window, etc.).
"""

from typing import List, Dict, Optional, Set
from uuid import UUID

from .models import (
    HierarchicalChunk,
    ExpandedChunk,
    ExpansionStrategy,
    HierarchyLevel,
    HierarchicalConfig
)


class ParentRetriever:
    """Expands chunks with parent context using configurable strategies.
    
    Takes small chunks retrieved from vector search and expands them with
    parent context to provide better information for LLM processing.
    """
    
    def __init__(
        self,
        chunk_repository,
        config: Optional[HierarchicalConfig] = None
    ):
        """Initialize the parent retriever.
        
        Args:
            chunk_repository: Repository for accessing chunk hierarchy
            config: Configuration for expansion
        """
        self.chunk_repo = chunk_repository
        self.config = config or HierarchicalConfig()
        self._token_estimator = lambda text: len(text.split())
    
    def expand_chunks(
        self,
        chunks: List[HierarchicalChunk],
        strategy: Optional[ExpansionStrategy] = None
    ) -> List[ExpandedChunk]:
        """Expand chunks with parent context.
        
        Args:
            chunks: List of chunks to expand
            strategy: Expansion strategy to use (defaults to config)
            
        Returns:
            List of expanded chunks with parent context
        """
        strategy = strategy or self.config.expansion_strategy
        
        expanded = []
        seen_parents: Set[str] = set()  # For deduplication
        
        for chunk in chunks:
            if strategy == ExpansionStrategy.IMMEDIATE_PARENT:
                expanded_chunk = self._expand_immediate_parent(chunk)
            elif strategy == ExpansionStrategy.FULL_SECTION:
                expanded_chunk = self._expand_full_section(chunk)
            elif strategy == ExpansionStrategy.WINDOW:
                expanded_chunk = self._expand_window(chunk)
            elif strategy == ExpansionStrategy.FULL_DOCUMENT:
                expanded_chunk = self._expand_full_document(chunk)
            elif strategy == ExpansionStrategy.ADAPTIVE:
                expanded_chunk = self._expand_adaptive(chunk)
            else:
                # No expansion
                expanded_chunk = ExpandedChunk(
                    original_chunk=chunk,
                    expanded_text=chunk.text,
                    expansion_strategy=strategy,
                    parent_chunks=[],
                    total_tokens=chunk.token_count
                )
            
            # Deduplicate: skip if we've already seen this parent
            parent_id = expanded_chunk.parent_chunks[0].chunk_id if expanded_chunk.parent_chunks else None
            if parent_id and parent_id in seen_parents:
                continue
            
            if parent_id:
                seen_parents.add(parent_id)
            
            expanded.append(expanded_chunk)
        
        return expanded
    
    def _expand_immediate_parent(self, chunk: HierarchicalChunk) -> ExpandedChunk:
        """Expand with immediate parent only.
        
        Args:
            chunk: The chunk to expand
            
        Returns:
            Expanded chunk with parent context
        """
        parent_chunks = []
        expanded_text = chunk.text
        
        if chunk.parent_chunk_id:
            parent = self.chunk_repo.get_by_id(UUID(chunk.parent_chunk_id))
            if parent:
                parent_chunk = self._db_chunk_to_hierarchical(parent)
                parent_chunks.append(parent_chunk)
                expanded_text = f"{parent_chunk.text}\n\n{chunk.text}"
        
        return ExpandedChunk(
            original_chunk=chunk,
            expanded_text=expanded_text,
            expansion_strategy=ExpansionStrategy.IMMEDIATE_PARENT,
            parent_chunks=parent_chunks,
            total_tokens=self._token_estimator(expanded_text)
        )
    
    def _expand_full_section(self, chunk: HierarchicalChunk) -> ExpandedChunk:
        """Expand with all ancestors up to section level.
        
        Args:
            chunk: The chunk to expand
            
        Returns:
            Expanded chunk with section context
        """
        parent_chunks = []
        
        if chunk.parent_chunk_id:
            # Get all ancestors
            ancestors = self.chunk_repo.get_ancestors(
                UUID(chunk.parent_chunk_id),
                max_depth=10
            )
            
            # Filter to section level and above
            for ancestor in ancestors:
                if ancestor.hierarchy_level <= HierarchyLevel.SECTION:
                    parent_chunks.append(self._db_chunk_to_hierarchical(ancestor))
        
        # Build expanded text from ancestors + chunk
        text_parts = [p.text for p in reversed(parent_chunks)]
        text_parts.append(chunk.text)
        expanded_text = "\n\n".join(text_parts)
        
        return ExpandedChunk(
            original_chunk=chunk,
            expanded_text=expanded_text,
            expansion_strategy=ExpansionStrategy.FULL_SECTION,
            parent_chunks=parent_chunks,
            total_tokens=self._token_estimator(expanded_text)
        )
    
    def _expand_window(self, chunk: HierarchicalChunk) -> ExpandedChunk:
        """Expand with N siblings before and after.
        
        Args:
            chunk: The chunk to expand
            
        Returns:
            Expanded chunk with sibling context
        """
        parent_chunks = []
        siblings = []
        
        if chunk.parent_chunk_id:
            # Get parent
            parent = self.chunk_repo.get_by_id(UUID(chunk.parent_chunk_id))
            if parent:
                parent_chunks.append(self._db_chunk_to_hierarchical(parent))
                
                # Get siblings
                all_siblings = self.chunk_repo.get_children(UUID(chunk.parent_chunk_id))
                
                # Find current position and get window
                current_pos = chunk.hierarchy_metadata.position_in_parent
                window_start = max(0, current_pos - self.config.window_size)
                window_end = min(
                    len(all_siblings),
                    current_pos + self.config.window_size + 1
                )
                
                siblings = [
                    self._db_chunk_to_hierarchical(s)
                    for s in all_siblings[window_start:window_end]
                ]
        
        # Build expanded text
        text_parts = [p.text for p in parent_chunks]
        text_parts.extend([s.text for s in siblings])
        expanded_text = "\n\n".join(text_parts)
        
        return ExpandedChunk(
            original_chunk=chunk,
            expanded_text=expanded_text,
            expansion_strategy=ExpansionStrategy.WINDOW,
            parent_chunks=parent_chunks,
            total_tokens=self._token_estimator(expanded_text)
        )
    
    def _expand_full_document(self, chunk: HierarchicalChunk) -> ExpandedChunk:
        """Expand with entire document context.
        
        Args:
            chunk: The chunk to expand
            
        Returns:
            Expanded chunk with full document context
        """
        parent_chunks = []
        
        # Get all ancestors up to root
        if chunk.parent_chunk_id:
            ancestors = self.chunk_repo.get_ancestors(
                UUID(chunk.parent_chunk_id),
                max_depth=10
            )
            parent_chunks = [self._db_chunk_to_hierarchical(a) for a in ancestors]
        
        # Get root (document level)
        root = next((p for p in parent_chunks if p.hierarchy_level == HierarchyLevel.DOCUMENT), None)
        expanded_text = root.text if root else chunk.text
        
        return ExpandedChunk(
            original_chunk=chunk,
            expanded_text=expanded_text,
            expansion_strategy=ExpansionStrategy.FULL_DOCUMENT,
            parent_chunks=parent_chunks,
            total_tokens=self._token_estimator(expanded_text)
        )
    
    def _expand_adaptive(self, chunk: HierarchicalChunk) -> ExpandedChunk:
        """Adaptively choose expansion strategy based on chunk characteristics.
        
        Args:
            chunk: The chunk to expand
            
        Returns:
            Expanded chunk with adaptively chosen context
        """
        # Decision logic:
        # - If chunk is very small (< 100 tokens), use FULL_SECTION
        # - If chunk is at paragraph level, use IMMEDIATE_PARENT
        # - If chunk is at sentence level, use WINDOW
        # - Otherwise, use IMMEDIATE_PARENT
        
        if chunk.token_count < 100:
            return self._expand_full_section(chunk)
        elif chunk.hierarchy_level == HierarchyLevel.PARAGRAPH:
            return self._expand_immediate_parent(chunk)
        elif chunk.hierarchy_level == HierarchyLevel.SENTENCE:
            return self._expand_window(chunk)
        else:
            return self._expand_immediate_parent(chunk)
    
    def _db_chunk_to_hierarchical(self, db_chunk) -> HierarchicalChunk:
        """Convert database chunk to hierarchical chunk.
        
        Args:
            db_chunk: Database Chunk model instance
            
        Returns:
            HierarchicalChunk instance
        """
        from .models import HierarchyMetadata
        
        # Extract hierarchy metadata from JSONB field
        h_meta = db_chunk.hierarchy_metadata or {}
        hierarchy_metadata = HierarchyMetadata(
            position_in_parent=h_meta.get('position_in_parent', 0),
            total_siblings=h_meta.get('total_siblings', 0),
            depth_from_root=h_meta.get('depth_from_root', 0)
        )
        
        return HierarchicalChunk(
            chunk_id=str(db_chunk.chunk_id),
            document_id=str(db_chunk.document_id),
            text=db_chunk.text,
            hierarchy_level=HierarchyLevel(db_chunk.hierarchy_level),
            hierarchy_metadata=hierarchy_metadata,
            parent_chunk_id=str(db_chunk.parent_chunk_id) if db_chunk.parent_chunk_id else None,
            token_count=len(db_chunk.text.split()),
            metadata=db_chunk.metadata_ or {},
            embedding=None  # Don't load embedding for parent chunks
        )
