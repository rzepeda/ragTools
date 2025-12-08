# Story 13.4: Implement Hierarchical Indexing

**Story ID:** 13.4
**Epic:** Epic 13 - Core Indexing Strategies Implementation
**Story Points:** 13
**Priority:** Medium
**Dependencies:** Epic 13.1 (Chunking)

---

## User Story

**As a** system
**I want** parent-child chunk relationships
**So that** retrieval can expand context

---

## Detailed Requirements

### Functional Requirements

1.  **Hierarchical Indexing Strategy**
    *   Implement `HierarchicalIndexing` class implementing `IIndexingStrategy`.
    *   Create a multi-level chunk hierarchy (Document -> Section -> Paragraph).
    *   Store parent-child relationships as metadata.
    *   Produce `CHUNKS`, `HIERARCHY`, and `DATABASE` capabilities.

2.  **Hierarchy Construction**
    *   Level 0: Full document.
    *   Level 1: Sections (split by headings).
    *   Level 2: Paragraphs.
    *   Support configurable max depth.

3.  **Metadata & Storage**
    *   Flatten the hierarchy for storage.
    *   Each chunk stores:
        *   `level`: The hierarchy level (0, 1, 2).
        *   `parent_id`: ID of the parent chunk.
        *   `path`: List of indices representing the path from root.
    *   Store chunks with this extended metadata in the database.

### Non-Functional Requirements

1.  **Performance**
    *   Target: <2s per 10k words.
    *   Efficient tree traversal and flattening.

2.  **Flexibility**
    *   Handle documents with varying structures (e.g., no headings).

---

## Acceptance Criteria

### AC1: Strategy Implementation
- [ ] `HierarchicalIndexing` class exists and implements `IIndexingStrategy`.
- [ ] `produces()` returns `{IndexCapability.CHUNKS, IndexCapability.HIERARCHY, IndexCapability.DATABASE}`.
- [ ] `requires_services()` returns `{ServiceDependency.DATABASE}`.

### AC2: Hierarchy Building
- [ ] Text is correctly split into hierarchical levels.
- [ ] Headings are detected for section splitting.
- [ ] Paragraphs are correctly identified.

### AC3: Metadata & Relationships
- [ ] Parent-child relationships are correctly established.
- [ ] `parent_id` and `level` metadata are accurate.
- [ ] Path metadata correctly reflects the hierarchy.

### AC4: Testing
- [ ] Unit tests for hierarchy construction logic.
- [ ] Integration tests verifying database storage of hierarchical data.
- [ ] Performance benchmarks meet targets.

---

## Technical Specifications

### Implementation

```python
from rag_factory.core.indexing import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.core.dependencies import ServiceDependency

class HierarchicalIndexing(IIndexingStrategy):
    """Creates hierarchical chunk relationships"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.CHUNKS,
            IndexCapability.HIERARCHY,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.DATABASE
        }
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Create hierarchical chunks with parent-child relationships.
        
        Hierarchy levels:
        - Level 0: Full document
        - Level 1: Sections (e.g., headings)
        - Level 2: Paragraphs
        - Level 3: Sentences (optional)
        """
        max_depth = self.config.get('max_depth', 2)
        all_chunks = []
        
        for doc in documents:
            # Create hierarchy
            hierarchy = self._build_hierarchy(doc.content, max_depth)
            
            # Flatten and assign IDs with relationships
            chunks = self._flatten_hierarchy(hierarchy, doc.id)
            all_chunks.extend(chunks)
        
        # Store chunks with hierarchy metadata
        await context.database.store_chunks_with_hierarchy(all_chunks)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'max_depth': max_depth,
                'avg_chunks_per_doc': len(all_chunks) / len(documents)
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )
    
    def _build_hierarchy(self, text: str, max_depth: int):
        """Build hierarchical structure from text"""
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
    
    def _flatten_hierarchy(self, hierarchy, doc_id):
        """Flatten hierarchy into chunks with parent references"""
        chunks = []
        
        def traverse(node, parent_id=None, path=[]):
            chunk_id = f"{doc_id}_{'_'.join(map(str, path))}"
            
            chunk = {
                'id': chunk_id,
                'document_id': doc_id,
                'text': node['text'],
                'level': node['level'],
                'parent_id': parent_id,
                'path': path.copy()
            }
            chunks.append(chunk)
            
            # Traverse children
            for i, child in enumerate(node['children']):
                traverse(child, chunk_id, path + [i])
        
        traverse(hierarchy)
        return chunks
    
    def _split_by_headings(self, text: str) -> list[str]:
        """Split text by markdown/HTML headings"""
        # Implementation
        pass
    
    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraph breaks"""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
```

### Technical Dependencies
- Database service with hierarchy support
