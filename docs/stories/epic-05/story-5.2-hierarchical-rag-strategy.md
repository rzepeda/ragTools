# Story 5.2: Implement Hierarchical RAG Strategy

**Story ID:** 5.2
**Epic:** Epic 5 - Agentic & Advanced Retrieval Strategies
**Story Points:** 13
**Sprint:** Sprint 6
**Priority:** High
**Dependencies:** Epic 4 (basic strategies), Epic 3 (embedding service, database schema)

---

## User Story

**As a** system
**I want** parent-child chunk relationships
**So that** I can search small but return large context

---

## Detailed Requirements

### Functional Requirements

1. **Parent-Child Chunk Relationships**
   - Define hierarchical metadata schema for chunks
   - Support multiple hierarchy levels (paragraph -> section -> document)
   - Store parent chunk ID references in child metadata
   - Store child chunk ID references in parent metadata
   - Support bidirectional relationship traversal
   - Handle orphaned chunks (no parent)
   - Preserve document structure (headings, sections, subsections)

2. **Multi-Level Chunking**
   - Small chunks for precise semantic search (paragraph level, 128-256 tokens)
   - Medium chunks for context sections (512-1024 tokens)
   - Large chunks for full document context (2048+ tokens)
   - Configurable chunk sizes per hierarchy level
   - Maintain text boundaries (don't split sentences/paragraphs)
   - Overlap handling at each hierarchy level
   - Preserve formatting and structure markers

3. **Search at Small Chunk Level**
   - Embed and index small chunks (highest granularity)
   - Perform semantic search on small chunks
   - Return top-k small chunks with highest similarity
   - Include small chunk metadata (position, parent references)
   - Support filtering by hierarchy level
   - Maintain search performance (<100ms for top-10)

4. **Parent Context Retrieval**
   - Given small chunk results, retrieve parent chunks
   - Support multiple expansion levels (immediate parent, grandparent, root)
   - Configurable expansion strategy (all parents, specific level, up to size limit)
   - Deduplicate parent chunks across multiple child matches
   - Preserve parent-child ordering
   - Include context window around small chunk within parent
   - Mark relevant small chunk position in parent context

5. **Hierarchy Levels Configuration**
   - Define hierarchy levels in configuration
   - Specify chunk size per level
   - Specify overlap per level
   - Define level names (e.g., "paragraph", "section", "document")
   - Support 2-5 hierarchy levels
   - Validate hierarchy consistency
   - Default configurations for common document types

6. **Expansion Strategies**
   - **Full Parent**: Return entire parent chunk
   - **Windowed Parent**: Return parent chunk with context window around child
   - **Multi-Level**: Return parent + grandparent
   - **Up to Size**: Expand until reaching token limit
   - **Sibling Context**: Include adjacent sibling chunks
   - **All Ancestors**: Return all parent chunks up to root
   - Configurable strategy selection per query

7. **Database Schema Updates**
   - Add `parent_chunk_id` field to chunks table
   - Add `hierarchy_level` field to chunks table (0=leaf, 1=section, 2=document)
   - Add `child_chunk_ids` JSONB field for children array
   - Add `position_in_parent` field for ordering
   - Add `hierarchy_path` field for full ancestry path
   - Create indexes on parent_chunk_id for fast lookups
   - Support recursive queries for ancestor retrieval

### Non-Functional Requirements

1. **Performance**
   - Small chunk search: <100ms for top-10 results
   - Parent chunk retrieval: <50ms for 10 parents
   - Hierarchy traversal: <10ms per level
   - Support 100K+ chunks per hierarchy level
   - Efficient batch parent retrieval

2. **Scalability**
   - Handle documents with 1000+ chunks
   - Support 5-level deep hierarchies
   - Efficient storage of relationship metadata
   - Minimize redundant text storage across levels

3. **Reliability**
   - Validate parent-child relationships on insert
   - Handle missing parent chunks gracefully
   - Detect and prevent circular references
   - Maintain referential integrity

4. **Maintainability**
   - Clear hierarchy configuration schema
   - Well-documented expansion strategies
   - Easy to add new hierarchy levels
   - Flexible chunking logic per level

5. **Observability**
   - Log hierarchy construction stats
   - Track expansion strategy usage
   - Monitor parent retrieval performance
   - Visualize hierarchy structure for debugging

---

## Acceptance Criteria

### AC1: Hierarchical Metadata Schema
- [ ] Parent-child relationships stored in chunk metadata
- [ ] Hierarchy level tracked for each chunk (0=leaf, higher=parent)
- [ ] Position in parent tracked for ordering
- [ ] Full ancestry path stored for fast traversal
- [ ] Metadata schema validated on chunk creation
- [ ] Bidirectional references maintained (parent->children, child->parent)

### AC2: Multi-Level Chunking
- [ ] Documents chunked at multiple granularities
- [ ] Small chunks (paragraphs) created for search
- [ ] Medium chunks (sections) created for context
- [ ] Large chunks (documents) created for full context
- [ ] Chunk sizes configurable per level
- [ ] Text boundaries respected (no sentence/paragraph splits)
- [ ] Overlap applied correctly at each level

### AC3: Small Chunk Search
- [ ] Embeddings generated for small chunks only
- [ ] Semantic search performed on small chunks
- [ ] Top-k small chunks returned with scores
- [ ] Small chunk metadata includes parent references
- [ ] Search completes in <100ms for top-10
- [ ] Results sorted by relevance score

### AC4: Parent Context Retrieval
- [ ] Parent chunks retrieved given child chunk IDs
- [ ] Multiple expansion strategies implemented
- [ ] Parent retrieval completes in <50ms for 10 chunks
- [ ] Deduplication of duplicate parents
- [ ] Child position highlighted in parent context
- [ ] Full ancestry retrievable when needed

### AC5: Hierarchy Configuration
- [ ] Configuration defines hierarchy levels
- [ ] Configuration specifies chunk sizes per level
- [ ] Configuration validates consistency
- [ ] Default configurations provided
- [ ] Configuration loaded from YAML/JSON
- [ ] Configuration errors reported clearly

### AC6: Expansion Strategies
- [ ] Full Parent strategy implemented
- [ ] Windowed Parent strategy implemented
- [ ] Multi-Level strategy implemented
- [ ] Up to Size strategy implemented
- [ ] Sibling Context strategy implemented
- [ ] Strategy selection configurable per query
- [ ] Strategy applied correctly in all cases

### AC7: Database Schema
- [ ] `parent_chunk_id` column added to chunks table
- [ ] `hierarchy_level` column added to chunks table
- [ ] `child_chunk_ids` JSONB column added
- [ ] `position_in_parent` column added
- [ ] `hierarchy_path` column added
- [ ] Indexes created on parent_chunk_id
- [ ] Migration scripts provided

### AC8: Testing
- [ ] Unit tests for hierarchy construction
- [ ] Unit tests for each expansion strategy
- [ ] Unit tests for parent retrieval
- [ ] Integration tests with real database
- [ ] Integration tests with real documents
- [ ] Performance benchmarks meet requirements

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── hierarchical/
│   │   ├── __init__.py
│   │   ├── strategy.py           # Main hierarchical strategy
│   │   ├── chunker.py            # Multi-level chunking logic
│   │   ├── hierarchy.py          # Hierarchy management
│   │   ├── expander.py           # Context expansion strategies
│   │   ├── config.py             # Hierarchical config
│   │   └── schema.py             # Metadata schemas
│
├── database/
│   ├── migrations/
│   │   └── 005_hierarchical_chunks.sql  # Schema migration
│
tests/
├── unit/
│   └── strategies/
│       └── hierarchical/
│           ├── test_strategy.py
│           ├── test_chunker.py
│           ├── test_hierarchy.py
│           ├── test_expander.py
│           └── test_config.py
│
├── integration/
│   └── strategies/
│       └── test_hierarchical_integration.py
```

### Dependencies
```python
# requirements.txt additions
pydantic>=2.0.0           # Schema validation
sqlalchemy>=2.0.0         # Database ORM
```

### Database Schema Migration
```sql
-- 005_hierarchical_chunks.sql
-- Add hierarchical relationship columns to chunks table

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_chunk_id UUID REFERENCES chunks(id);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS hierarchy_level INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS child_chunk_ids JSONB DEFAULT '[]'::jsonb;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS position_in_parent INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS hierarchy_path JSONB DEFAULT '[]'::jsonb;

-- Create indexes for fast parent lookups
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy_level ON chunks(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy_path ON chunks USING gin(hierarchy_path);

-- Add comments
COMMENT ON COLUMN chunks.parent_chunk_id IS 'Reference to parent chunk in hierarchy';
COMMENT ON COLUMN chunks.hierarchy_level IS 'Level in hierarchy (0=leaf/small, higher=parent/large)';
COMMENT ON COLUMN chunks.child_chunk_ids IS 'Array of child chunk IDs';
COMMENT ON COLUMN chunks.position_in_parent IS 'Position index within parent chunk';
COMMENT ON COLUMN chunks.hierarchy_path IS 'Array of ancestor IDs from root to this chunk';

-- Function to validate no circular references
CREATE OR REPLACE FUNCTION validate_chunk_hierarchy()
RETURNS TRIGGER AS $$
BEGIN
    -- Prevent self-reference
    IF NEW.parent_chunk_id = NEW.id THEN
        RAISE EXCEPTION 'Chunk cannot be its own parent';
    END IF;

    -- Prevent circular references by checking if new parent is in ancestry path
    IF NEW.parent_chunk_id IS NOT NULL AND
       NEW.parent_chunk_id = ANY(SELECT jsonb_array_elements_text(NEW.hierarchy_path)::uuid) THEN
        RAISE EXCEPTION 'Circular reference detected in chunk hierarchy';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for validation
DROP TRIGGER IF EXISTS validate_chunk_hierarchy_trigger ON chunks;
CREATE TRIGGER validate_chunk_hierarchy_trigger
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION validate_chunk_hierarchy();
```

### Hierarchical Configuration
```python
# rag_factory/strategies/hierarchical/config.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class ExpansionStrategy(str, Enum):
    """Available context expansion strategies."""
    FULL_PARENT = "full_parent"                    # Return entire parent chunk
    WINDOWED_PARENT = "windowed_parent"            # Parent with window around child
    MULTI_LEVEL = "multi_level"                    # Parent + grandparent
    UP_TO_SIZE = "up_to_size"                      # Expand until token limit
    SIBLING_CONTEXT = "sibling_context"            # Include adjacent siblings
    ALL_ANCESTORS = "all_ancestors"                # All parents to root

class HierarchyLevel(BaseModel):
    """Configuration for a single hierarchy level."""
    name: str                                       # e.g., "paragraph", "section", "document"
    level: int                                      # 0=leaf (smallest), higher=parent
    chunk_size: int                                 # Tokens per chunk at this level
    chunk_overlap: int = 0                          # Token overlap between chunks
    description: str = ""                           # Human-readable description

    @validator('level')
    def validate_level(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Hierarchy level must be between 0 and 10")
        return v

    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 1 or v > 10000:
            raise ValueError("Chunk size must be between 1 and 10000 tokens")
        return v

class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical RAG strategy."""

    # Hierarchy levels (must be in order from smallest to largest)
    levels: List[HierarchyLevel] = Field(
        default_factory=lambda: [
            HierarchyLevel(name="paragraph", level=0, chunk_size=200, chunk_overlap=20),
            HierarchyLevel(name="section", level=1, chunk_size=800, chunk_overlap=100),
            HierarchyLevel(name="document", level=2, chunk_size=3000, chunk_overlap=0)
        ]
    )

    # Search configuration
    search_level: int = 0                           # Which level to search (0=smallest)

    # Expansion configuration
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.FULL_PARENT
    expansion_window_tokens: int = 200              # For WINDOWED_PARENT strategy
    expansion_max_tokens: int = 2000                # For UP_TO_SIZE strategy
    expansion_levels: int = 1                       # How many levels to expand (for MULTI_LEVEL)

    # Retrieval configuration
    top_k_search: int = 5                           # Top-k small chunks to retrieve
    deduplicate_parents: bool = True                # Remove duplicate parent chunks
    include_child_position: bool = True             # Mark child position in parent

    # Performance configuration
    batch_parent_retrieval: bool = True             # Batch retrieve parents
    cache_parent_chunks: bool = True                # Cache frequently accessed parents

    @validator('levels')
    def validate_levels_order(cls, v):
        """Ensure levels are ordered by size."""
        if len(v) < 2:
            raise ValueError("At least 2 hierarchy levels required")

        for i in range(len(v) - 1):
            if v[i].level >= v[i+1].level:
                raise ValueError("Levels must be ordered by level number")
            if v[i].chunk_size >= v[i+1].chunk_size:
                raise ValueError("Levels must be ordered by chunk size (small to large)")

        return v

    @validator('search_level')
    def validate_search_level(cls, v, values):
        """Ensure search_level is valid."""
        if 'levels' in values and v >= len(values['levels']):
            raise ValueError(f"search_level {v} must be < number of levels")
        return v

# Default configurations for common document types
DEFAULT_CONFIGS = {
    "standard": HierarchicalConfig(
        levels=[
            HierarchyLevel(name="paragraph", level=0, chunk_size=200, chunk_overlap=20),
            HierarchyLevel(name="section", level=1, chunk_size=800, chunk_overlap=100),
            HierarchyLevel(name="document", level=2, chunk_size=3000, chunk_overlap=0)
        ],
        expansion_strategy=ExpansionStrategy.FULL_PARENT
    ),

    "fine_grained": HierarchicalConfig(
        levels=[
            HierarchyLevel(name="sentence", level=0, chunk_size=50, chunk_overlap=5),
            HierarchyLevel(name="paragraph", level=1, chunk_size=200, chunk_overlap=20),
            HierarchyLevel(name="section", level=2, chunk_size=800, chunk_overlap=100),
            HierarchyLevel(name="chapter", level=3, chunk_size=2000, chunk_overlap=200),
            HierarchyLevel(name="document", level=4, chunk_size=5000, chunk_overlap=0)
        ],
        expansion_strategy=ExpansionStrategy.MULTI_LEVEL,
        expansion_levels=2
    ),

    "code_documentation": HierarchicalConfig(
        levels=[
            HierarchyLevel(name="function", level=0, chunk_size=150, chunk_overlap=10),
            HierarchyLevel(name="class", level=1, chunk_size=600, chunk_overlap=50),
            HierarchyLevel(name="module", level=2, chunk_size=2000, chunk_overlap=100),
            HierarchyLevel(name="package", level=3, chunk_size=5000, chunk_overlap=0)
        ],
        expansion_strategy=ExpansionStrategy.SIBLING_CONTEXT
    )
}
```

### Hierarchical Chunker
```python
# rag_factory/strategies/hierarchical/chunker.py
from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
from .config import HierarchicalConfig, HierarchyLevel

logger = logging.getLogger(__name__)

class ChunkNode:
    """Represents a chunk in the hierarchy."""

    def __init__(
        self,
        text: str,
        level: int,
        document_id: str,
        chunk_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        position: int = 0
    ):
        self.id = chunk_id or str(uuid.uuid4())
        self.text = text
        self.level = level
        self.document_id = document_id
        self.parent_id = parent_id
        self.position = position
        self.children: List[ChunkNode] = []
        self.hierarchy_path: List[str] = []

    def add_child(self, child: 'ChunkNode'):
        """Add a child chunk."""
        child.parent_id = self.id
        child.hierarchy_path = self.hierarchy_path + [self.id]
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "level": self.level,
            "document_id": self.document_id,
            "parent_chunk_id": self.parent_id,
            "position_in_parent": self.position,
            "hierarchy_path": self.hierarchy_path,
            "child_chunk_ids": [child.id for child in self.children],
            "metadata": {
                "hierarchy_level": self.level,
                "has_children": len(self.children) > 0,
                "num_children": len(self.children)
            }
        }


class HierarchicalChunker:
    """
    Chunks documents into multiple hierarchy levels.

    Creates a tree structure where:
    - Leaf nodes are small chunks (for search)
    - Parent nodes are larger chunks (for context)
    - Each chunk references its parent and children
    """

    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.levels = sorted(config.levels, key=lambda x: x.level)

    def chunk_document(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ChunkNode]:
        """
        Chunk a document into hierarchical levels.

        Args:
            text: Full document text
            document_id: Document identifier
            metadata: Additional document metadata

        Returns:
            List of all chunks (flattened tree)
        """
        logger.info(f"Chunking document {document_id} into {len(self.levels)} levels")

        # Start with the largest level (root document)
        root_level = self.levels[-1]
        root_chunks = self._chunk_at_level(text, root_level, document_id)

        # Build hierarchy from top down
        all_chunks = []
        self._build_hierarchy(root_chunks, 0, all_chunks)

        logger.info(
            f"Created {len(all_chunks)} total chunks for document {document_id} "
            f"({sum(1 for c in all_chunks if c.level == 0)} leaf chunks)"
        )

        return all_chunks

    def _build_hierarchy(
        self,
        parent_chunks: List[ChunkNode],
        level_idx: int,
        all_chunks: List[ChunkNode]
    ):
        """Recursively build hierarchy from top down."""
        # Add parent chunks to collection
        all_chunks.extend(parent_chunks)

        # If we've reached the smallest level, stop
        if level_idx >= len(self.levels) - 1:
            return

        # Get next level (one level smaller)
        child_level = self.levels[-(level_idx + 2)]

        # Chunk each parent into children
        for parent_idx, parent in enumerate(parent_chunks):
            child_chunks = self._chunk_at_level(
                parent.text,
                child_level,
                parent.document_id,
                parent_id=parent.id
            )

            # Set position and add to parent
            for child_idx, child in enumerate(child_chunks):
                child.position = child_idx
                parent.add_child(child)

            # Recursively process children
            self._build_hierarchy(child_chunks, level_idx + 1, all_chunks)

    def _chunk_at_level(
        self,
        text: str,
        level: HierarchyLevel,
        document_id: str,
        parent_id: Optional[str] = None
    ) -> List[ChunkNode]:
        """
        Chunk text at a specific hierarchy level.

        Args:
            text: Text to chunk
            level: Hierarchy level configuration
            document_id: Document identifier
            parent_id: Parent chunk ID (if any)

        Returns:
            List of chunks at this level
        """
        # Simple token-based chunking
        # In production, use more sophisticated chunking (sentence boundaries, etc.)
        tokens = text.split()
        chunks = []

        chunk_size = level.chunk_size
        overlap = level.chunk_overlap

        start = 0
        position = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunk = ChunkNode(
                text=chunk_text,
                level=level.level,
                document_id=document_id,
                parent_id=parent_id,
                position=position
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap if overlap > 0 else end
            position += 1

            # Prevent infinite loop
            if start >= len(tokens):
                break

        return chunks
```

### Context Expansion Strategies
```python
# rag_factory/strategies/hierarchical/expander.py
from typing import List, Dict, Any, Optional, Set
import logging
from .config import ExpansionStrategy, HierarchicalConfig

logger = logging.getLogger(__name__)

class ContextExpander:
    """
    Expands small chunk results to include parent context.

    Implements various expansion strategies to retrieve
    parent chunks given small chunk search results.
    """

    def __init__(self, config: HierarchicalConfig, chunk_repository):
        self.config = config
        self.chunk_repository = chunk_repository

    def expand(
        self,
        small_chunks: List[Dict[str, Any]],
        strategy: Optional[ExpansionStrategy] = None
    ) -> List[Dict[str, Any]]:
        """
        Expand small chunks to include parent context.

        Args:
            small_chunks: List of small chunk search results
            strategy: Override default expansion strategy

        Returns:
            List of expanded chunks with parent context
        """
        strategy = strategy or self.config.expansion_strategy

        logger.info(f"Expanding {len(small_chunks)} chunks using {strategy}")

        if strategy == ExpansionStrategy.FULL_PARENT:
            return self._expand_full_parent(small_chunks)
        elif strategy == ExpansionStrategy.WINDOWED_PARENT:
            return self._expand_windowed_parent(small_chunks)
        elif strategy == ExpansionStrategy.MULTI_LEVEL:
            return self._expand_multi_level(small_chunks)
        elif strategy == ExpansionStrategy.UP_TO_SIZE:
            return self._expand_up_to_size(small_chunks)
        elif strategy == ExpansionStrategy.SIBLING_CONTEXT:
            return self._expand_sibling_context(small_chunks)
        elif strategy == ExpansionStrategy.ALL_ANCESTORS:
            return self._expand_all_ancestors(small_chunks)
        else:
            logger.warning(f"Unknown strategy {strategy}, returning small chunks")
            return small_chunks

    def _expand_full_parent(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve entire parent chunk for each small chunk."""
        parent_ids = [
            chunk.get("parent_chunk_id")
            for chunk in small_chunks
            if chunk.get("parent_chunk_id")
        ]

        if not parent_ids:
            return small_chunks

        # Batch retrieve parent chunks
        parent_chunks = self.chunk_repository.get_chunks_by_ids(parent_ids)

        # Deduplicate if configured
        if self.config.deduplicate_parents:
            parent_chunks = self._deduplicate_chunks(parent_chunks)

        # Mark child positions in parents if configured
        if self.config.include_child_position:
            parent_chunks = self._mark_child_positions(parent_chunks, small_chunks)

        return parent_chunks

    def _expand_windowed_parent(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve parent chunk with context window around small chunk."""
        parent_chunks = self._expand_full_parent(small_chunks)

        # Create windowed versions
        windowed_chunks = []
        for parent in parent_chunks:
            # Find matching small chunks in this parent
            child_matches = [
                small for small in small_chunks
                if small.get("parent_chunk_id") == parent["id"]
            ]

            if not child_matches:
                windowed_chunks.append(parent)
                continue

            # Extract window around each child
            for child in child_matches:
                windowed = self._extract_window(
                    parent,
                    child,
                    window_size=self.config.expansion_window_tokens
                )
                windowed_chunks.append(windowed)

        return self._deduplicate_chunks(windowed_chunks)

    def _expand_multi_level(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve multiple levels of parents (parent, grandparent, etc.)."""
        expanded_chunks = []
        levels_to_expand = self.config.expansion_levels

        current_chunks = small_chunks
        for level in range(levels_to_expand):
            # Get parents of current chunks
            parent_ids = [
                chunk.get("parent_chunk_id")
                for chunk in current_chunks
                if chunk.get("parent_chunk_id")
            ]

            if not parent_ids:
                break

            parents = self.chunk_repository.get_chunks_by_ids(parent_ids)
            expanded_chunks.extend(parents)

            # Use parents as input for next level
            current_chunks = parents

        return self._deduplicate_chunks(expanded_chunks)

    def _expand_up_to_size(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand parents until reaching token limit."""
        expanded_chunks = []
        total_tokens = 0
        max_tokens = self.config.expansion_max_tokens

        # Start with immediate parents
        parent_ids = list(set(
            chunk.get("parent_chunk_id")
            for chunk in small_chunks
            if chunk.get("parent_chunk_id")
        ))

        while parent_ids and total_tokens < max_tokens:
            parents = self.chunk_repository.get_chunks_by_ids(parent_ids)

            for parent in parents:
                # Estimate tokens (rough approximation)
                chunk_tokens = len(parent["text"].split())

                if total_tokens + chunk_tokens <= max_tokens:
                    expanded_chunks.append(parent)
                    total_tokens += chunk_tokens
                else:
                    break

            # Get next level parents
            parent_ids = list(set(
                chunk.get("parent_chunk_id")
                for chunk in parents
                if chunk.get("parent_chunk_id")
            ))

        return expanded_chunks

    def _expand_sibling_context(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Include adjacent sibling chunks for additional context."""
        expanded_chunks = []

        for chunk in small_chunks:
            parent_id = chunk.get("parent_chunk_id")
            position = chunk.get("position_in_parent", 0)

            if not parent_id:
                expanded_chunks.append(chunk)
                continue

            # Get parent to find siblings
            parent = self.chunk_repository.get_chunk_by_id(parent_id)
            if not parent:
                expanded_chunks.append(chunk)
                continue

            # Get all children of parent (siblings)
            sibling_ids = parent.get("child_chunk_ids", [])
            siblings = self.chunk_repository.get_chunks_by_ids(sibling_ids)

            # Include previous and next siblings
            for sibling in siblings:
                sibling_pos = sibling.get("position_in_parent", 0)
                if abs(sibling_pos - position) <= 1:
                    expanded_chunks.append(sibling)

        return self._deduplicate_chunks(expanded_chunks)

    def _expand_all_ancestors(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve all ancestor chunks up to root document."""
        expanded_chunks = []

        for chunk in small_chunks:
            hierarchy_path = chunk.get("hierarchy_path", [])

            if hierarchy_path:
                # Get all ancestors
                ancestors = self.chunk_repository.get_chunks_by_ids(hierarchy_path)
                expanded_chunks.extend(ancestors)

            expanded_chunks.append(chunk)

        return self._deduplicate_chunks(expanded_chunks)

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks by ID."""
        seen: Set[str] = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks

    def _mark_child_positions(
        self,
        parent_chunks: List[Dict[str, Any]],
        child_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mark positions of children in parent chunks."""
        for parent in parent_chunks:
            matching_children = [
                child for child in child_chunks
                if child.get("parent_chunk_id") == parent["id"]
            ]

            parent["metadata"] = parent.get("metadata", {})
            parent["metadata"]["matched_children"] = [
                {
                    "id": child["id"],
                    "position": child.get("position_in_parent", 0),
                    "score": child.get("score", 0)
                }
                for child in matching_children
            ]

        return parent_chunks

    def _extract_window(
        self,
        parent: Dict[str, Any],
        child: Dict[str, Any],
        window_size: int
    ) -> Dict[str, Any]:
        """Extract a window of tokens around child in parent."""
        # Simplified implementation
        # In production, use proper text positioning
        parent_text = parent["text"]
        child_text = child["text"]

        # Find child in parent
        child_pos = parent_text.find(child_text)

        if child_pos == -1:
            # Child not found in parent, return full parent
            return parent

        # Extract window
        tokens = parent_text.split()
        window_tokens = tokens[:window_size]
        window_text = " ".join(window_tokens)

        windowed_chunk = parent.copy()
        windowed_chunk["text"] = window_text
        windowed_chunk["metadata"] = windowed_chunk.get("metadata", {})
        windowed_chunk["metadata"]["windowed"] = True
        windowed_chunk["metadata"]["window_size"] = window_size

        return windowed_chunk
```

### Hierarchical Strategy
```python
# rag_factory/strategies/hierarchical/strategy.py
from typing import List, Dict, Any, Optional
import logging
from ..base import RAGStrategy
from .config import HierarchicalConfig, ExpansionStrategy
from .chunker import HierarchicalChunker
from .expander import ContextExpander

logger = logging.getLogger(__name__)

class HierarchicalRAGStrategy(RAGStrategy):
    """
    Hierarchical RAG strategy with parent-child chunk relationships.

    Search small chunks for precision, retrieve parent chunks for context.

    Flow:
    1. Chunk documents into multiple hierarchy levels
    2. Embed and index only small chunks (leaf level)
    3. Search small chunks for semantic similarity
    4. Expand results to include parent context
    5. Return parent chunks with marked child positions
    """

    def __init__(
        self,
        embedding_service,
        chunk_repository,
        config: Optional[HierarchicalConfig] = None
    ):
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository
        self.config = config or HierarchicalConfig()

        self.chunker = HierarchicalChunker(self.config)
        self.expander = ContextExpander(self.config, chunk_repository)

        logger.info(
            f"Initialized HierarchicalRAGStrategy with {len(self.config.levels)} levels"
        )

    def prepare_data(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare documents with hierarchical chunking.

        Args:
            documents: List of documents to chunk

        Returns:
            Dict with prepared chunks and stats
        """
        logger.info(f"Preparing {len(documents)} documents with hierarchical chunking")

        all_chunks = []
        stats = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "chunks_per_level": {},
            "leaf_chunks": 0
        }

        for doc in documents:
            # Chunk document into hierarchy
            doc_chunks = self.chunker.chunk_document(
                text=doc["text"],
                document_id=doc["id"],
                metadata=doc.get("metadata", {})
            )

            all_chunks.extend(doc_chunks)

            # Track stats
            for chunk in doc_chunks:
                level = chunk.level
                stats["chunks_per_level"][level] = stats["chunks_per_level"].get(level, 0) + 1

                if level == 0:
                    stats["leaf_chunks"] += 1

        stats["total_chunks"] = len(all_chunks)

        # Store all chunks in repository
        self.chunk_repository.store_chunks([chunk.to_dict() for chunk in all_chunks])

        # Embed only leaf chunks (level 0)
        leaf_chunks = [chunk for chunk in all_chunks if chunk.level == 0]
        leaf_chunk_dicts = [chunk.to_dict() for chunk in leaf_chunks]

        # Generate embeddings
        texts = [chunk["text"] for chunk in leaf_chunk_dicts]
        embeddings = self.embedding_service.embed_batch(texts)

        # Store embeddings
        for chunk, embedding in zip(leaf_chunk_dicts, embeddings):
            chunk["embedding"] = embedding

        self.chunk_repository.store_embeddings(leaf_chunk_dicts)

        logger.info(
            f"Prepared {stats['total_chunks']} chunks "
            f"({stats['leaf_chunks']} leaf chunks embedded)"
        )

        return stats

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expansion_strategy: Optional[ExpansionStrategy] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hierarchical search.

        Args:
            query: Search query
            top_k: Number of results
            expansion_strategy: Override default expansion strategy
            **kwargs: Additional parameters

        Returns:
            List of expanded chunks with parent context
        """
        logger.info(f"Hierarchical retrieval for query: {query}")

        # 1. Embed query
        query_embedding = self.embedding_service.embed(query)

        # 2. Search small chunks (leaf level)
        search_level = self.config.search_level
        small_chunks = self.chunk_repository.search_by_embedding(
            embedding=query_embedding,
            top_k=top_k,
            level=search_level
        )

        logger.info(f"Found {len(small_chunks)} small chunks at level {search_level}")

        # 3. Expand to parent context
        expanded_chunks = self.expander.expand(
            small_chunks,
            strategy=expansion_strategy
        )

        logger.info(f"Expanded to {len(expanded_chunks)} chunks")

        # 4. Add strategy metadata
        for chunk in expanded_chunks:
            chunk["strategy"] = "hierarchical"
            chunk["hierarchy_metadata"] = {
                "search_level": search_level,
                "expansion_strategy": (expansion_strategy or self.config.expansion_strategy).value,
                "num_small_chunks": len(small_chunks)
            }

        return expanded_chunks[:top_k]

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def description(self) -> str:
        return "Search small chunks, retrieve parent context"
```

---

## Unit Tests

### Test File Locations
- `tests/unit/strategies/hierarchical/test_strategy.py`
- `tests/unit/strategies/hierarchical/test_chunker.py`
- `tests/unit/strategies/hierarchical/test_expander.py`
- `tests/unit/strategies/hierarchical/test_config.py`

### Test Cases

#### TC5.2.1: Configuration Tests
```python
import pytest
from rag_factory.strategies.hierarchical.config import (
    HierarchicalConfig, HierarchyLevel, ExpansionStrategy
)

def test_hierarchy_level_validation():
    """Test hierarchy level parameter validation."""
    # Valid level
    level = HierarchyLevel(name="paragraph", level=0, chunk_size=200)
    assert level.chunk_size == 200

    # Invalid level number
    with pytest.raises(ValueError):
        HierarchyLevel(name="test", level=-1, chunk_size=100)

    # Invalid chunk size
    with pytest.raises(ValueError):
        HierarchyLevel(name="test", level=0, chunk_size=-100)

def test_hierarchical_config_defaults():
    """Test default configuration."""
    config = HierarchicalConfig()

    assert len(config.levels) >= 2
    assert config.search_level == 0
    assert config.expansion_strategy == ExpansionStrategy.FULL_PARENT
    assert config.deduplicate_parents == True

def test_hierarchical_config_level_ordering():
    """Test that levels must be ordered correctly."""
    # Valid ordering
    config = HierarchicalConfig(
        levels=[
            HierarchyLevel(name="small", level=0, chunk_size=100),
            HierarchyLevel(name="large", level=1, chunk_size=500)
        ]
    )
    assert len(config.levels) == 2

    # Invalid ordering (wrong level numbers)
    with pytest.raises(ValueError):
        HierarchicalConfig(
            levels=[
                HierarchyLevel(name="large", level=1, chunk_size=500),
                HierarchyLevel(name="small", level=0, chunk_size=100)
            ]
        )

    # Invalid ordering (wrong chunk sizes)
    with pytest.raises(ValueError):
        HierarchicalConfig(
            levels=[
                HierarchyLevel(name="large", level=0, chunk_size=500),
                HierarchyLevel(name="small", level=1, chunk_size=100)
            ]
        )

def test_config_validation():
    """Test configuration validation."""
    config = HierarchicalConfig(
        levels=[
            HierarchyLevel(name="small", level=0, chunk_size=100),
            HierarchyLevel(name="large", level=1, chunk_size=500)
        ],
        search_level=0
    )

    # Invalid search level
    with pytest.raises(ValueError):
        HierarchicalConfig(
            levels=config.levels,
            search_level=5  # > number of levels
        )
```

#### TC5.2.2: Chunker Tests
```python
import pytest
from rag_factory.strategies.hierarchical.chunker import (
    HierarchicalChunker, ChunkNode
)
from rag_factory.strategies.hierarchical.config import (
    HierarchicalConfig, HierarchyLevel
)

@pytest.fixture
def simple_config():
    return HierarchicalConfig(
        levels=[
            HierarchyLevel(name="small", level=0, chunk_size=10, chunk_overlap=2),
            HierarchyLevel(name="large", level=1, chunk_size=30, chunk_overlap=5)
        ]
    )

def test_chunk_node_creation():
    """Test ChunkNode initialization."""
    node = ChunkNode(
        text="Test text",
        level=0,
        document_id="doc1",
        position=0
    )

    assert node.text == "Test text"
    assert node.level == 0
    assert node.document_id == "doc1"
    assert len(node.children) == 0

def test_chunk_node_add_child():
    """Test adding children to chunk node."""
    parent = ChunkNode("Parent", level=1, document_id="doc1")
    child = ChunkNode("Child", level=0, document_id="doc1")

    parent.add_child(child)

    assert len(parent.children) == 1
    assert child.parent_id == parent.id
    assert parent.id in child.hierarchy_path

def test_chunk_node_to_dict():
    """Test chunk node serialization."""
    node = ChunkNode("Test", level=0, document_id="doc1", position=5)
    node_dict = node.to_dict()

    assert node_dict["text"] == "Test"
    assert node_dict["level"] == 0
    assert node_dict["position_in_parent"] == 5
    assert "metadata" in node_dict
    assert "child_chunk_ids" in node_dict

def test_hierarchical_chunker_initialization(simple_config):
    """Test chunker initialization."""
    chunker = HierarchicalChunker(simple_config)

    assert len(chunker.levels) == 2
    assert chunker.levels[0].level == 0

def test_chunk_document_creates_hierarchy(simple_config):
    """Test document chunking creates proper hierarchy."""
    chunker = HierarchicalChunker(simple_config)

    # Simple text with ~50 tokens
    text = " ".join([f"word{i}" for i in range(50)])

    chunks = chunker.chunk_document(text, "doc1")

    # Should have both levels
    assert len(chunks) > 0
    levels_present = set(chunk.level for chunk in chunks)
    assert 0 in levels_present  # Small chunks
    assert 1 in levels_present  # Large chunks

def test_chunk_hierarchy_relationships(simple_config):
    """Test parent-child relationships are correct."""
    chunker = HierarchicalChunker(simple_config)

    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunker.chunk_document(text, "doc1")

    # Find a small chunk with a parent
    small_chunks = [c for c in chunks if c.level == 0]
    assert len(small_chunks) > 0

    for small_chunk in small_chunks:
        if small_chunk.parent_id:
            # Find parent
            parent = next(c for c in chunks if c.id == small_chunk.parent_id)

            # Parent should have higher level
            assert parent.level > small_chunk.level

            # Parent should reference child
            assert small_chunk.id in [child.id for child in parent.children]

def test_chunker_respects_chunk_size(simple_config):
    """Test chunks respect configured size limits."""
    chunker = HierarchicalChunker(simple_config)

    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunker.chunk_document(text, "doc1")

    for chunk in chunks:
        level_config = next(l for l in chunker.levels if l.level == chunk.level)
        chunk_tokens = len(chunk.text.split())

        # Allow some flexibility (<=20% over)
        assert chunk_tokens <= level_config.chunk_size * 1.2
```

#### TC5.2.3: Expander Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.strategies.hierarchical.expander import ContextExpander
from rag_factory.strategies.hierarchical.config import (
    HierarchicalConfig, ExpansionStrategy
)

@pytest.fixture
def mock_chunk_repository():
    repo = Mock()

    # Mock parent chunks
    repo.get_chunks_by_ids.return_value = [
        {
            "id": "parent1",
            "text": "This is the full parent chunk text with more context",
            "level": 1,
            "child_chunk_ids": ["child1", "child2"]
        }
    ]

    repo.get_chunk_by_id.return_value = {
        "id": "parent1",
        "text": "Parent chunk",
        "level": 1,
        "child_chunk_ids": ["child1", "child2"]
    }

    return repo

@pytest.fixture
def simple_config():
    return HierarchicalConfig()

@pytest.fixture
def small_chunks():
    return [
        {
            "id": "child1",
            "text": "Small chunk 1",
            "level": 0,
            "parent_chunk_id": "parent1",
            "position_in_parent": 0,
            "score": 0.9
        },
        {
            "id": "child2",
            "text": "Small chunk 2",
            "level": 0,
            "parent_chunk_id": "parent1",
            "position_in_parent": 1,
            "score": 0.85
        }
    ]

def test_expander_initialization(simple_config, mock_chunk_repository):
    """Test expander initialization."""
    expander = ContextExpander(simple_config, mock_chunk_repository)

    assert expander.config == simple_config
    assert expander.chunk_repository == mock_chunk_repository

def test_expand_full_parent(simple_config, mock_chunk_repository, small_chunks):
    """Test full parent expansion strategy."""
    expander = ContextExpander(simple_config, mock_chunk_repository)

    expanded = expander.expand(small_chunks, ExpansionStrategy.FULL_PARENT)

    assert len(expanded) >= 1
    mock_chunk_repository.get_chunks_by_ids.assert_called_once()

def test_expand_deduplicates_parents(simple_config, mock_chunk_repository, small_chunks):
    """Test that duplicate parents are removed."""
    simple_config.deduplicate_parents = True
    expander = ContextExpander(simple_config, mock_chunk_repository)

    # Both small chunks have same parent
    expanded = expander.expand(small_chunks, ExpansionStrategy.FULL_PARENT)

    # Should return only one parent chunk
    parent_ids = [chunk["id"] for chunk in expanded]
    assert len(parent_ids) == len(set(parent_ids))

def test_expand_marks_child_positions(simple_config, mock_chunk_repository, small_chunks):
    """Test that child positions are marked in parents."""
    simple_config.include_child_position = True
    expander = ContextExpander(simple_config, mock_chunk_repository)

    expanded = expander.expand(small_chunks, ExpansionStrategy.FULL_PARENT)

    # Check parent has metadata about children
    if expanded:
        parent = expanded[0]
        assert "metadata" in parent
        assert "matched_children" in parent["metadata"]

def test_expand_multi_level(simple_config, mock_chunk_repository, small_chunks):
    """Test multi-level expansion retrieves grandparents."""
    simple_config.expansion_levels = 2
    expander = ContextExpander(simple_config, mock_chunk_repository)

    # Mock grandparent retrieval
    mock_chunk_repository.get_chunks_by_ids.side_effect = [
        [{"id": "parent1", "level": 1, "parent_chunk_id": "grandparent1"}],
        [{"id": "grandparent1", "level": 2, "parent_chunk_id": None}]
    ]

    expanded = expander.expand(small_chunks, ExpansionStrategy.MULTI_LEVEL)

    # Should call repository twice (parent, grandparent)
    assert mock_chunk_repository.get_chunks_by_ids.call_count == 2

def test_expand_handles_no_parent(simple_config, mock_chunk_repository):
    """Test expansion handles chunks without parents."""
    expander = ContextExpander(simple_config, mock_chunk_repository)

    orphan_chunks = [
        {"id": "orphan1", "text": "Orphan chunk", "level": 0}
    ]

    expanded = expander.expand(orphan_chunks, ExpansionStrategy.FULL_PARENT)

    # Should return original chunks
    assert len(expanded) >= 0
```

#### TC5.2.4: Strategy Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.strategies.hierarchical.strategy import HierarchicalRAGStrategy
from rag_factory.strategies.hierarchical.config import HierarchicalConfig

@pytest.fixture
def mock_embedding_service():
    service = Mock()
    service.embed.return_value = [0.1] * 384
    service.embed_batch.return_value = [[0.1] * 384 for _ in range(5)]
    return service

@pytest.fixture
def mock_chunk_repository():
    repo = Mock()
    repo.store_chunks.return_value = None
    repo.store_embeddings.return_value = None
    repo.search_by_embedding.return_value = [
        {"id": "chunk1", "text": "Result 1", "level": 0, "score": 0.9, "parent_chunk_id": "parent1"},
        {"id": "chunk2", "text": "Result 2", "level": 0, "score": 0.85, "parent_chunk_id": "parent1"}
    ]
    repo.get_chunks_by_ids.return_value = [
        {"id": "parent1", "text": "Parent chunk", "level": 1}
    ]
    return repo

def test_strategy_initialization(mock_embedding_service, mock_chunk_repository):
    """Test strategy initialization."""
    strategy = HierarchicalRAGStrategy(
        mock_embedding_service,
        mock_chunk_repository
    )

    assert strategy.name == "hierarchical"
    assert strategy.chunker is not None
    assert strategy.expander is not None

def test_strategy_prepare_data(mock_embedding_service, mock_chunk_repository):
    """Test data preparation with hierarchical chunking."""
    strategy = HierarchicalRAGStrategy(
        mock_embedding_service,
        mock_chunk_repository
    )

    documents = [
        {"id": "doc1", "text": " ".join([f"word{i}" for i in range(100)])}
    ]

    stats = strategy.prepare_data(documents)

    assert "total_documents" in stats
    assert stats["total_documents"] == 1
    assert "total_chunks" in stats
    assert "leaf_chunks" in stats

    # Should store chunks and embeddings
    mock_chunk_repository.store_chunks.assert_called_once()
    mock_chunk_repository.store_embeddings.assert_called_once()

def test_strategy_retrieve(mock_embedding_service, mock_chunk_repository):
    """Test retrieval with expansion."""
    strategy = HierarchicalRAGStrategy(
        mock_embedding_service,
        mock_chunk_repository
    )

    results = strategy.retrieve("test query", top_k=5)

    assert len(results) > 0
    mock_embedding_service.embed.assert_called_once_with("test query")
    mock_chunk_repository.search_by_embedding.assert_called_once()

def test_strategy_adds_metadata(mock_embedding_service, mock_chunk_repository):
    """Test strategy adds hierarchy metadata to results."""
    strategy = HierarchicalRAGStrategy(
        mock_embedding_service,
        mock_chunk_repository
    )

    results = strategy.retrieve("test query", top_k=5)

    for result in results:
        assert result["strategy"] == "hierarchical"
        assert "hierarchy_metadata" in result
        assert "search_level" in result["hierarchy_metadata"]
        assert "expansion_strategy" in result["hierarchy_metadata"]
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_hierarchical_integration.py`

### Test Scenarios

#### IS5.2.1: End-to-End Hierarchical Retrieval
```python
@pytest.mark.integration
def test_hierarchical_retrieval_workflow(test_db):
    """Test complete hierarchical retrieval workflow."""
    from rag_factory.strategies.hierarchical import HierarchicalRAGStrategy
    from rag_factory.strategies.hierarchical.config import HierarchicalConfig

    # Setup services
    embedding_service = Mock()  # Use real service in production
    chunk_repository = Mock()   # Use real repository in production

    # Create strategy
    config = HierarchicalConfig()
    strategy = HierarchicalRAGStrategy(embedding_service, chunk_repository, config)

    # Prepare documents
    documents = [
        {
            "id": "doc1",
            "text": "This is a test document. " * 100  # Long document
        }
    ]

    stats = strategy.prepare_data(documents)

    assert stats["total_chunks"] > 0
    assert stats["leaf_chunks"] > 0

    # Retrieve
    results = strategy.retrieve("test query", top_k=5)

    assert len(results) > 0
    assert all("strategy" in r for r in results)

@pytest.mark.integration
def test_hierarchy_consistency(test_db):
    """Test that hierarchy relationships are consistent."""
    # Create hierarchy
    # Verify all parent references are valid
    # Verify all child references are valid
    # Verify no circular references
    pass

@pytest.mark.integration
def test_expansion_strategies_comparison(test_db):
    """Test different expansion strategies on same data."""
    # Create test data
    # Try each expansion strategy
    # Compare results (coverage, context size, relevance)
    pass
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_hierarchical_performance.py

@pytest.mark.benchmark
def test_small_chunk_search_latency():
    """Test small chunk search is <100ms."""
    # Measure search time on 100K chunks
    # Assert <100ms
    pass

@pytest.mark.benchmark
def test_parent_retrieval_latency():
    """Test parent retrieval is <50ms."""
    # Measure parent retrieval for 10 chunks
    # Assert <50ms
    pass

@pytest.mark.benchmark
def test_hierarchy_construction_throughput():
    """Test chunking throughput."""
    # Measure chunks/second for large documents
    # Assert reasonable throughput
    pass
```

---

## Definition of Done

- [ ] HierarchicalConfig implemented with validation
- [ ] HierarchicalChunker creates multi-level chunks
- [ ] ChunkNode properly manages parent-child relationships
- [ ] ContextExpander implements all expansion strategies
- [ ] HierarchicalRAGStrategy retrieves and expands correctly
- [ ] Database schema migration completed
- [ ] Parent-child relationships stored in database
- [ ] Hierarchy path tracking implemented
- [ ] All expansion strategies working
- [ ] Deduplication implemented
- [ ] Child position marking implemented
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Database indexes created
- [ ] Configuration validation working
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Database Migration

```bash
# Run migration to add hierarchical columns
psql -U postgres -d rag_db -f rag_factory/database/migrations/005_hierarchical_chunks.sql
```

### Configuration

```yaml
# config.yaml
strategies:
  hierarchical:
    enabled: true

    # Hierarchy levels (small to large)
    levels:
      - name: paragraph
        level: 0
        chunk_size: 200
        chunk_overlap: 20

      - name: section
        level: 1
        chunk_size: 800
        chunk_overlap: 100

      - name: document
        level: 2
        chunk_size: 3000
        chunk_overlap: 0

    # Search configuration
    search_level: 0  # Search at paragraph level

    # Expansion configuration
    expansion_strategy: full_parent
    expansion_window_tokens: 200
    expansion_max_tokens: 2000
    expansion_levels: 1

    # Retrieval configuration
    top_k_search: 5
    deduplicate_parents: true
    include_child_position: true
```

### Usage Example

```python
from rag_factory.strategies.hierarchical import HierarchicalRAGStrategy
from rag_factory.strategies.hierarchical.config import (
    HierarchicalConfig, ExpansionStrategy
)

# Create configuration
config = HierarchicalConfig(
    expansion_strategy=ExpansionStrategy.FULL_PARENT,
    top_k_search=5
)

# Create strategy
strategy = HierarchicalRAGStrategy(
    embedding_service=embedding_service,
    chunk_repository=chunk_repository,
    config=config
)

# Prepare documents
stats = strategy.prepare_data(documents)
print(f"Created {stats['total_chunks']} chunks across {len(config.levels)} levels")

# Retrieve with expansion
results = strategy.retrieve(
    "What are the main features?",
    top_k=5,
    expansion_strategy=ExpansionStrategy.MULTI_LEVEL
)

# View results
for result in results:
    print(f"Level {result['level']}: {result['text'][:100]}...")
    if "matched_children" in result.get("metadata", {}):
        print(f"  Matched children: {result['metadata']['matched_children']}")
```

---

## Notes for Developers

1. **Hierarchy Depth**: Start with 2-3 levels. More levels add complexity and storage overhead.

2. **Chunk Sizes**: Choose sizes based on use case:
   - Code: smaller chunks (50-200 tokens)
   - Documentation: medium chunks (200-500 tokens)
   - Books: larger chunks (500-1000 tokens)

3. **Search Level**: Always search the smallest level (0) for maximum precision.

4. **Expansion Strategy**:
   - Use FULL_PARENT for general purpose
   - Use MULTI_LEVEL for comprehensive context
   - Use UP_TO_SIZE for token-limited scenarios

5. **Database Performance**: Index parent_chunk_id and hierarchy_path for fast traversal.

6. **Storage Efficiency**: Large chunks can be stored without embeddings to save space.

7. **Testing**: Test with real documents that have clear structure (headings, sections).

8. **Debugging**: Use hierarchy_path field to visualize chunk relationships.

9. **Circular References**: Database trigger prevents circular parent-child relationships.

10. **Deduplication**: Always enable when multiple small chunks share the same parent.
