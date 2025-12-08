# Story 12.1: Define Capability Enums and Models

**Story ID:** 12.1
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 5
**Priority:** High
**Dependencies:** Epic 11 (Dependency Injection)

---

## User Story

**As a** developer
**I want** capability enums and result models
**So that** strategies can declare what they produce/require

---

## Detailed Requirements

### Functional Requirements

1.  **Capability Enum Definition**
    -   Define `IndexCapability` enum with all capability types
    -   Include storage types (VECTORS, KEYWORDS, GRAPH, FULL_DOCUMENT)
    -   Include structure types (CHUNKS, HIERARCHY, LATE_CHUNKS)
    -   Include storage backends (IN_MEMORY, FILE_BACKED, DATABASE)
    -   Include enrichment types (CONTEXTUAL, METADATA)

2.  **Indexing Result Model**
    -   Create `IndexingResult` dataclass
    -   Include capabilities set
    -   Include metadata dictionary
    -   Include document and chunk counts
    -   Implement `has_capability` method
    -   Implement `is_compatible_with` method

3.  **Validation Result Model**
    -   Create `ValidationResult` dataclass
    -   Include validity boolean
    -   Include missing capabilities set
    -   Include missing services set (from Epic 11)
    -   Include message string
    -   Include suggestions list

### Non-Functional Requirements

1.  **Maintainability**
    -   Clear documentation for each capability
    -   Extensible enum for future capabilities
    -   Type hints for all models

2.  **Usability**
    -   Helpful string representations (`__repr__`) for debugging
    -   Easy-to-use compatibility checks

---

## Acceptance Criteria

### AC1: Capability Enum
- [ ] `IndexCapability` enum defined with all required members
- [ ] Docstrings provided for each capability
- [ ] Enum supports auto-values or unique string values

### AC2: Indexing Result
- [ ] `IndexingResult` class implemented
- [ ] `has_capability` returns correct boolean
- [ ] `is_compatible_with` correctly checks subset relationship
- [ ] `__repr__` provides readable summary

### AC3: Validation Result
- [ ] `ValidationResult` class implemented
- [ ] Supports missing capabilities and services
- [ ] `__repr__` provides clear error/success message

### AC4: Testing
- [ ] Unit tests for `IndexCapability` enum members
- [ ] Unit tests for `IndexingResult` methods
- [ ] Unit tests for `ValidationResult` formatting

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── capabilities.py      # New file for capabilities and models
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_capabilities.py
│   └── ...
```

### Code Definition

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Set, Dict, List

class IndexCapability(Enum):
    """Capabilities that an indexing strategy can produce"""
    
    # Storage types - what kind of searchable data is created
    VECTORS = auto()              # Vector embeddings stored in database
    KEYWORDS = auto()             # Keyword/BM25 index created
    GRAPH = auto()                # Knowledge graph with entities/relationships
    FULL_DOCUMENT = auto()        # Complete documents stored as-is
    
    # Structure types - how documents are organized
    CHUNKS = auto()               # Documents split into chunks
    HIERARCHY = auto()            # Parent-child relationships between chunks
    LATE_CHUNKS = auto()          # Late chunking (embed-then-chunk) applied
    
    # Storage backends - where data is persisted
    IN_MEMORY = auto()            # Data stored in memory only (for testing)
    FILE_BACKED = auto()          # Data persisted to files
    DATABASE = auto()             # Data persisted to database
    
    # Enrichment types - additional processing applied
    CONTEXTUAL = auto()           # Chunks have contextual descriptions
    METADATA = auto()             # Rich metadata extracted and indexed

@dataclass
class IndexingResult:
    """Result of an indexing operation"""
    
    capabilities: Set[IndexCapability]
    metadata: Dict
    document_count: int
    chunk_count: int
    
    def has_capability(self, cap: IndexCapability) -> bool:
        """Check if specific capability is present"""
        return cap in self.capabilities
    
    def is_compatible_with(self, requirements: Set[IndexCapability]) -> bool:
        """Check if capabilities satisfy requirements"""
        return requirements.issubset(self.capabilities)
    
    def __repr__(self) -> str:
        caps = [c.name for c in self.capabilities]
        return f"IndexingResult(capabilities={{{', '.join(caps)}}}, docs={self.document_count}, chunks={self.chunk_count})"

@dataclass
class ValidationResult:
    """Result of compatibility validation"""
    
    is_valid: bool
    missing_capabilities: Set[IndexCapability]
    missing_services: Set['ServiceDependency']  # From Epic 11
    message: str
    suggestions: List[str]
    
    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(valid=True)"
        
        issues = []
        if self.missing_capabilities:
            caps = [c.name for c in self.missing_capabilities]
            issues.append(f"capabilities: {', '.join(caps)}")
        if self.missing_services:
            svcs = [s.name for s in self.missing_services]
            issues.append(f"services: {', '.join(svcs)}")
        
        return f"ValidationResult(valid=False, missing: {'; '.join(issues)})"
```
