# Story 12.6: Implement Factory Validation with Consistency Checking

**Story ID:** 12.6
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 12.4, Story 12.5, Epic 11 (Story 11.6)

---

## User Story

**As a** developer
**I want** the factory to validate pipeline compatibility and warn about inconsistencies
**So that** invalid combinations are caught early and suspicious patterns are flagged

---

## Detailed Requirements

### Functional Requirements

1.  **Factory Validation Methods**
    -   Add `validate_compatibility()` to `RAGFactory`
    -   Check if indexing capabilities satisfy retrieval requirements
    -   Add `validate_pipeline()` to `RAGFactory`
    -   Perform full validation (capabilities + services)
    -   Integrate `ConsistencyChecker` (from Epic 11) for strategy consistency checks

2.  **Automatic Strategy Selection**
    -   Add `auto_select_retrieval()` method
    -   Select compatible retrieval strategies based on indexing pipeline capabilities
    -   Prioritize preferred strategies if provided

3.  **Validation Reporting**
    -   Return detailed `ValidationResult`
    -   Include specific missing capabilities/services
    -   Provide helpful suggestions for fixing issues
    -   Log warnings for non-critical inconsistencies

### Non-Functional Requirements

1.  **Usability**
    -   Clear and actionable error messages
    -   Helpful suggestions for resolution

2.  **Reliability**
    -   Accurate validation logic
    -   No false positives/negatives for critical errors

---

## Acceptance Criteria

### AC1: Compatibility Validation
- [ ] `validate_compatibility` correctly identifies missing capabilities
- [ ] Returns valid result when requirements are met
- [ ] Runs consistency checks and logs warnings

### AC2: Full Pipeline Validation
- [ ] `validate_pipeline` checks both capabilities and services
- [ ] Returns comprehensive `ValidationResult`

### AC3: Auto-Selection
- [ ] `auto_select_retrieval` picks compatible strategies
- [ ] Respects preferences if possible
- [ ] Warns if no compatible strategy found

### AC4: Testing
- [ ] Unit tests for validation with various combinations
- [ ] Test with consistent and inconsistent strategies
- [ ] Test auto-selection logic

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── factory.py           # Existing file (update RAGFactory)
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_factory_validation.py # New test file
│   └── ...
```

### Code Definition

```python
from typing import List, Optional
from .capabilities import IndexCapability, ValidationResult
from .pipeline import IndexingPipeline, RetrievalPipeline
from .consistency import ConsistencyChecker # From Epic 11

class RAGFactory:
    # ... existing DI code ...
    
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.consistency_checker = ConsistencyChecker()
        self._indexing_registry = {}
        self._retrieval_registry = {}
    
    def validate_compatibility(
        self,
        indexing_pipeline: IndexingPipeline,
        retrieval_pipeline: RetrievalPipeline
    ) -> ValidationResult:
        """
        Validate capability compatibility between pipelines.
        
        Also checks consistency of strategies (warns, doesn't fail).
        
        Args:
            indexing_pipeline: Indexing pipeline to validate
            retrieval_pipeline: Retrieval pipeline to validate
            
        Returns:
            ValidationResult indicating compatibility
        """
        # Check consistency of strategies (warnings only)
        for strategy in indexing_pipeline.strategies:
            self.consistency_checker.check_and_log(strategy, "indexing")
        
        for strategy in retrieval_pipeline.strategies:
            self.consistency_checker.check_and_log(strategy, "retrieval")
        
        # Check capability compatibility (can fail)
        capabilities = indexing_pipeline.get_capabilities()
        requirements = retrieval_pipeline.get_requirements()
        
        missing_caps = requirements - capabilities
        
        if missing_caps:
            suggestions = self._generate_suggestions(missing_caps)
            return ValidationResult(
                is_valid=False,
                missing_capabilities=missing_caps,
                missing_services=set(),
                message=f"Missing capabilities: {[c.name for c in missing_caps]}",
                suggestions=suggestions
            )
        
        return ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipelines are compatible",
            suggestions=[]
        )
    
    def validate_pipeline(
        self,
        indexing_pipeline: IndexingPipeline,
        retrieval_pipeline: RetrievalPipeline
    ) -> ValidationResult:
        """
        Full validation: capabilities AND services.
        
        Also runs consistency checks (warns about suspicious patterns).
        
        Args:
            indexing_pipeline: Indexing pipeline
            retrieval_pipeline: Retrieval pipeline
            
        Returns:
            Complete ValidationResult
        """
        # Check capabilities (includes consistency checking)
        cap_validation = self.validate_compatibility(indexing_pipeline, retrieval_pipeline)
        if not cap_validation.is_valid:
            return cap_validation
        
        # Check services (already validated at pipeline creation, but double-check)
        service_reqs = retrieval_pipeline.get_service_requirements()
        is_valid, missing = self.dependencies.validate_for_strategy(service_reqs)
        
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                missing_capabilities=set(),
                missing_services=set(missing),
                message=f"Missing services: {[s.name for s in missing]}",
                suggestions=[]
            )
        
        return ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipeline fully valid (capabilities and services)",
            suggestions=[]
        )
```
