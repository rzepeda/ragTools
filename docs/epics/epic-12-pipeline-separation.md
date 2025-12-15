# Epic 12: Indexing/Retrieval Pipeline Separation & Capability System

**Epic Goal:** Separate indexing and retrieval into independent pipelines with capability-based validation, enabling experimental RAG approaches beyond traditional vector search.

**Epic Story Points Total:** 55

**Dependencies:** Epic 11 (Dependency Injection - must be complete first)

**Status:** Ready for implementation after Epic 11

---

## Background

Current RAG implementations assume:
- All strategies use vector embeddings
- Indexing and retrieval are coupled
- Single indexing approach per system

This epic breaks these assumptions by:
1. Separating indexing (write path) from retrieval (read path)
2. Using capability enums to declare what indexing produces and retrieval requires
3. Validating compatibility at pipeline creation time
4. Enabling experimental approaches (graph-only, keyword-only, no-storage, etc.)

---

## Core Concepts

### Two Independent Pipelines

```
┌─────────────────────┐         ┌─────────────────────┐
│  Indexing Pipeline  │  ────>  │ Retrieval Pipeline  │
│   (Write Path)      │         │   (Read Path)       │
└─────────────────────┘         └─────────────────────┘
         │                               │
         │                               │
    produces                         requires
  capabilities                     capabilities
         │                               │
         └──────── validation ───────────┘
```

### Capability-Based Matching

**Indexing strategies declare:** "I produce {VECTORS, CHUNKS, DATABASE}"  
**Retrieval strategies declare:** "I require {VECTORS}"  
**System validates:** retrieval requirements ⊆ indexing capabilities

---

## Story 12.1: Define Capability Enums and Models

**As a** developer  
**I want** capability enums and result models  
**So that** strategies can declare what they produce/require

**Acceptance Criteria:**
- Create `IndexCapability` enum with all capability types
- Create `IndexingResult` class to hold capabilities + metadata
- Create `ValidationResult` class for compatibility checks
- Comprehensive documentation for all capabilities
- Unit tests for enum operations
- Examples showing capability usage

**Capability System:**

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

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
    
    capabilities: set[IndexCapability]
    metadata: dict
    document_count: int
    chunk_count: int
    
    def has_capability(self, cap: IndexCapability) -> bool:
        """Check if specific capability is present"""
        return cap in self.capabilities
    
    def is_compatible_with(self, requirements: set[IndexCapability]) -> bool:
        """Check if capabilities satisfy requirements"""
        return requirements.issubset(self.capabilities)
    
    def __repr__(self) -> str:
        caps = [c.name for c in self.capabilities]
        return f"IndexingResult(capabilities={{{', '.join(caps)}}}, docs={self.document_count}, chunks={self.chunk_count})"

@dataclass
class ValidationResult:
    """Result of compatibility validation"""
    
    is_valid: bool
    missing_capabilities: set[IndexCapability]
    missing_services: set['ServiceDependency']  # From Epic 11
    message: str
    suggestions: list[str]
    
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

**Story Points:** 5

---

## Story 12.2: Create IIndexingStrategy Interface

**As a** developer  
**I want** a separate interface for indexing strategies  
**So that** indexing is independent from retrieval

**Acceptance Criteria:**
- Define `IIndexingStrategy` abstract base class
- Add `produces()` method returning capability set
- Add `requires_services()` method (from Epic 11)
- Add `process()` method for document indexing
- Create `IndexingContext` for shared state
- Documentation with interface contract
- Example implementations

**Interface Definition:**

```python
from abc import ABC, abstractmethod

class IndexingContext:
    """Shared context for indexing operations"""
    
    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: dict = None
    ):
        self.database = database_service
        self.config = config or {}
        self.metrics = {}  # For tracking performance

class IIndexingStrategy(ABC):
    """Interface for document indexing strategies"""
    
    def __init__(
        self,
        config: dict,
        dependencies: 'StrategyDependencies'  # From Epic 11
    ):
        """
        Initialize indexing strategy.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services (validated at creation)
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        from rag_factory.core.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())
    
    @abstractmethod
    def produces(self) -> set[IndexCapability]:
        """
        Declare what capabilities this strategy produces.
        
        Returns:
            Set of IndexCapability enums
            
        Example:
            return {IndexCapability.VECTORS, IndexCapability.CHUNKS, IndexCapability.DATABASE}
        """
        pass
    
    @abstractmethod
    def requires_services(self) -> set['ServiceDependency']:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of ServiceDependency enums (from Epic 11)
            
        Example:
            return {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
        """
        pass
    
    @abstractmethod
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Process documents for indexing.
        
        Args:
            documents: Documents to index
            context: Shared indexing context
            
        Returns:
            IndexingResult with capabilities produced and metadata
        """
        pass
```

**Example Implementation:**

```python
class VectorEmbeddingIndexing(IIndexingStrategy):
    """Creates vector embeddings and stores in database"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }
    
    async def process(
        self,
        documents: list[Document],
        context: IndexingContext
    ) -> IndexingResult:
        # Use injected embedding service
        texts = [doc.content for doc in documents]
        embeddings = await self.deps.embedding_service.embed_batch(texts)
        
        # Store in database via context
        await context.database.store_embeddings(embeddings)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                "embedding_model": self.config.get("model", "default"),
                "dimension": len(embeddings[0])
            },
            document_count=len(documents),
            chunk_count=len(embeddings)
        )
```

**Story Points:** 8

---

## Story 12.3: Create IRetrievalStrategy Interface

**As a** developer  
**I want** retrieval strategies to declare capability requirements  
**So that** compatibility can be validated

**Acceptance Criteria:**
- Define `IRetrievalStrategy` abstract base class
- Add `requires()` method returning required capabilities
- Add `requires_services()` method (from Epic 11)
- Add `retrieve()` method for document retrieval
- Remove `prepare_data()` method (moved to indexing)
- Create `RetrievalContext` for shared state
- Documentation with interface contract
- Example implementations

**Interface Definition:**

```python
class RetrievalContext:
    """Shared context for retrieval operations"""
    
    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: dict = None
    ):
        self.database = database_service
        self.config = config or {}
        self.metrics = {}

class IRetrievalStrategy(ABC):
    """Interface for document retrieval strategies"""
    
    def __init__(
        self,
        config: dict,
        dependencies: 'StrategyDependencies'
    ):
        """
        Initialize retrieval strategy.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        from rag_factory.core.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())
    
    @abstractmethod
    def requires(self) -> set[IndexCapability]:
        """
        Declare what index capabilities this strategy requires.
        
        Returns:
            Set of required IndexCapability enums
            
        Example:
            return {IndexCapability.VECTORS, IndexCapability.CHUNKS}
        """
        pass
    
    @abstractmethod
    def requires_services(self) -> set['ServiceDependency']:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of ServiceDependency enums
            
        Example:
            return {ServiceDependency.LLM, ServiceDependency.DATABASE}
        """
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> list['Chunk']:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User query
            context: Shared retrieval context
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        pass
```

**Example Implementation:**

```python
class RerankingRetrieval(IRetrievalStrategy):
    """Two-step retrieval with reranking"""
    
    def requires(self) -> set[IndexCapability]:
        return {
            IndexCapability.VECTORS,
            IndexCapability.CHUNKS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.RERANKER,
            ServiceDependency.DATABASE
        }
    
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> list[Chunk]:
        # Step 1: Get many candidates using vectors
        query_embedding = await self.deps.embedding_service.embed(query)
        candidates = await context.database.search_chunks(
            query_embedding,
            top_k=top_k * 5  # Get 5x more for reranking
        )
        
        # Step 2: Rerank using specialized model
        texts = [c.content for c in candidates]
        reranked_indices = await self.deps.reranker_service.rerank(
            query,
            texts,
            top_k=top_k
        )
        
        # Return top reranked results
        return [candidates[idx] for idx, score in reranked_indices]
```

**Story Points:** 8

---

## Story 12.4: Implement IndexingPipeline

**As a** developer  
**I want** a pipeline that executes indexing strategies in sequence  
**So that** multiple indexing strategies can be combined

**Acceptance Criteria:**
- Create `IndexingPipeline` class
- Execute strategies in order
- Aggregate capabilities from all strategies
- Return combined `IndexingResult`
- Support async execution
- Track metrics and timing
- Handle errors gracefully
- Unit tests with multiple strategies

**Implementation:**

```python
class IndexingPipeline:
    """Pipeline for executing indexing strategies"""
    
    def __init__(
        self,
        strategies: list[IIndexingStrategy],
        context: IndexingContext
    ):
        """
        Create indexing pipeline.
        
        Args:
            strategies: Ordered list of indexing strategies
            context: Shared indexing context
        """
        self.strategies = strategies
        self.context = context
        self._last_result: Optional[IndexingResult] = None
    
    def get_capabilities(self) -> set[IndexCapability]:
        """
        Get combined capabilities from all strategies.
        
        Returns:
            Union of all strategy capabilities
        """
        if self._last_result:
            return self._last_result.capabilities
        
        # Return declared capabilities if not executed yet
        all_caps = set()
        for strategy in self.strategies:
            all_caps.update(strategy.produces())
        return all_caps
    
    async def index(
        self,
        documents: list['Document']
    ) -> IndexingResult:
        """
        Execute indexing pipeline.
        
        Args:
            documents: Documents to index
            
        Returns:
            Combined IndexingResult from all strategies
        """
        all_capabilities = set()
        all_metadata = {}
        total_chunks = 0
        
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            
            # Execute strategy
            result = await strategy.process(documents, self.context)
            
            # Aggregate results
            all_capabilities.update(result.capabilities)
            all_metadata[strategy_name] = result.metadata
            total_chunks = max(total_chunks, result.chunk_count)
        
        self._last_result = IndexingResult(
            capabilities=all_capabilities,
            metadata=all_metadata,
            document_count=len(documents),
            chunk_count=total_chunks
        )
        
        return self._last_result
```

**Usage Example:**

```python
async def example():
    # Create strategies
    chunking = ContextAwareChunking(config={}, dependencies=deps)
    embedding = VectorEmbeddingIndexing(config={}, dependencies=deps)
    keywords = KeywordIndexing(config={}, dependencies=deps)

    # Create pipeline
    pipeline = IndexingPipeline(
        strategies=[chunking, embedding, keywords],
        context=IndexingContext(database_service=db)
    )

    # Index documents
    result = await pipeline.index(documents)
    print(result)
    # Output: IndexingResult(capabilities={CHUNKS, VECTORS, KEYWORDS, DATABASE}, docs=100, chunks=450)

```

**Story Points:** 13

---

## Story 12.5: Implement RetrievalPipeline

**As a** developer  
**I want** a pipeline that executes retrieval strategies in sequence  
**So that** multiple retrieval strategies can be chained

**Acceptance Criteria:**
- Create `RetrievalPipeline` class
- Execute strategies in order (each operates on previous results)
- Get combined requirements from all strategies
- Support async execution
- Track metrics and timing
- Handle errors gracefully
- Unit tests with multiple strategies

**Implementation:**

```python
class RetrievalPipeline:
    """Pipeline for executing retrieval strategies"""
    
    def __init__(
        self,
        strategies: list[IRetrievalStrategy],
        context: RetrievalContext
    ):
        """
        Create retrieval pipeline.
        
        Args:
            strategies: Ordered list of retrieval strategies
            context: Shared retrieval context
        """
        self.strategies = strategies
        self.context = context
    
    def get_requirements(self) -> set[IndexCapability]:
        """
        Get combined requirements from all strategies.
        
        Returns:
            Union of all strategy requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires())
        return all_reqs
    
    def get_service_requirements(self) -> set['ServiceDependency']:
        """
        Get combined service requirements from all strategies.
        
        Returns:
            Union of all service requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires_services())
        return all_reqs
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> list['Chunk']:
        """
        Execute retrieval pipeline.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Retrieved chunks after all strategies applied
        """
        current_query = query
        results = None
        
        for strategy in self.strategies:
            # Each strategy processes the query and/or refines results
            results = await strategy.retrieve(current_query, self.context, top_k)
        
        return results
```

**Usage Example:**

```python
async def example():
    # Create strategies
    expansion = QueryExpansionRetrieval(config={}, dependencies=deps)
    reranking = RerankingRetrieval(config={}, dependencies=deps)

    # Create pipeline
    pipeline = RetrievalPipeline(
        strategies=[expansion, reranking],
        context=RetrievalContext(database_service=db)
    )

    # Retrieve
    results = await pipeline.retrieve("What are the action items?", top_k=5)

```

**Story Points:** 8

---

## Story 12.6: Implement Factory Validation with Consistency Checking

**As a** developer  
**I want** the factory to validate pipeline compatibility and warn about inconsistencies  
**So that** invalid combinations are caught early and suspicious patterns are flagged

**Acceptance Criteria:**
- Add `validate_compatibility()` to factory (checks capabilities)
- Add `validate_pipeline()` to factory (checks capabilities + services)
- Add `auto_select_retrieval()` for automatic strategy selection
- Integrate `ConsistencyChecker` from Epic 11 into validation flow
- Clear error messages with suggestions
- Warning messages for inconsistencies (don't block)
- Unit tests with valid and invalid combinations
- Unit tests with consistent and inconsistent strategies
- Documentation with validation examples

**Factory Methods:**

```python
class RAGFactory:
    # ... existing DI code from Epic 11 ...
    
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.consistency_checker = ConsistencyChecker()  # From Epic 11 Story 11.6
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
    
    def auto_select_retrieval(
        self,
        indexing_pipeline: IndexingPipeline,
        preferred_strategies: list[str] = None
    ) -> RetrievalPipeline:
        """
        Automatically select compatible retrieval strategies.
        
        Checks both capability requirements and service availability.
        Also warns about inconsistencies in selected strategies.
        
        Args:
            indexing_pipeline: Indexing pipeline to match against
            preferred_strategies: Optional list of preferred strategy names
            
        Returns:
            Compatible retrieval pipeline
            
        Raises:
            ValueError: If no compatible strategies found
        """
        capabilities = indexing_pipeline.get_capabilities()
        compatible = []
        
        # Find compatible strategies
        for name, strategy_class in self._retrieval_registry.items():
            # Check if preferred
            if preferred_strategies and name not in preferred_strategies:
                continue
            
            # Check capability requirements
            temp = strategy_class.__new__(strategy_class)
            
            # Check consistency (warn only)
            warnings = self.consistency_checker.check_retrieval_strategy(temp)
            for warning in warnings:
                logger.warning(warning)
            
            required_caps = temp.requires()
            if not required_caps.issubset(capabilities):
                continue
            
            # Check service requirements
            required_svcs = temp.requires_services()
            is_valid, _ = self.dependencies.validate_for_strategy(required_svcs)
            if not is_valid:
                continue
            
            compatible.append(name)
        
        if not compatible:
            raise ValueError(
                f"No compatible retrieval strategies found for capabilities: "
                f"{[c.name for c in capabilities]}"
            )
        
        # Create pipeline with compatible strategies
        return self.create_retrieval_pipeline(
            compatible,
            [{}] * len(compatible)
        )
    
    def _generate_suggestions(
        self,
        missing_caps: set[IndexCapability]
    ) -> list[str]:
        """Generate helpful suggestions for missing capabilities"""
        suggestions = []
        
        if IndexCapability.VECTORS in missing_caps:
            suggestions.append("Add VectorEmbeddingIndexing to indexing pipeline")
        if IndexCapability.KEYWORDS in missing_caps:
            suggestions.append("Add KeywordIndexing to indexing pipeline")
        if IndexCapability.GRAPH in missing_caps:
            suggestions.append("Add KnowledgeGraphIndexing to indexing pipeline")
        if IndexCapability.CHUNKS in missing_caps:
            suggestions.append("Add ContextAwareChunking to indexing pipeline")
        if IndexCapability.HIERARCHY in missing_caps:
            suggestions.append("Add HierarchicalIndexing to indexing pipeline")
        
        return suggestions
```

**Validation with Consistency Checking Example:**

```python
async def example():
    # Create pipelines
    indexing = factory.create_indexing_pipeline(
        ["context_aware_chunking", "vector_embedding"],
        [{}, {}]
    )

    retrieval = factory.create_retrieval_pipeline(
        ["reranking"],
        [{}]
    )

    # Validate (includes consistency checks)
    validation = factory.validate_pipeline(indexing, retrieval)

    # Console output might show:
    # ⚠️  SomeStrategy: Produces VECTORS but doesn't require EMBEDDING service.
    #     This is unusual unless loading pre-computed embeddings.
    # ✅ Pipeline fully valid (capabilities and services)

    if validation.is_valid:
        # Use pipelines
        result = await indexing.index(documents)
        chunks = await retrieval.retrieve(query)

```

**CLI Integration (extends Story 8.5.1):**

```bash
# Validate pipelines with consistency checking
$ rag-factory validate-pipeline --indexing chunking,embedding --retrieval reranking

Running consistency checks...
  ✅ context_aware_chunking: Consistent
  ✅ vector_embedding: Consistent
  ✅ reranking: Consistent

Validating pipeline compatibility...
  Indexing capabilities: {CHUNKS, VECTORS, DATABASE}
  Retrieval requirements: {VECTORS, CHUNKS}
  ✅ Compatible

Validating service availability...
  ✅ All required services available

Result: Pipeline is valid ✅

# Check with inconsistent strategy
$ rag-factory validate-pipeline --indexing weird_strategy --retrieval reranking

Running consistency checks...
  ⚠️  weird_strategy: Produces VECTORS but doesn't require EMBEDDING service
     This is unusual unless loading pre-computed embeddings.
  ✅ reranking: Consistent

Validating pipeline compatibility...
  ❌ Missing capabilities: CHUNKS
  Suggestion: Add ContextAwareChunking to indexing pipeline

Result: Pipeline is invalid ❌
```

**Testing Example:**

```python
def test_validation_with_consistency():
    # Create factory with checker
    factory = RAGFactory(dependencies=deps)
    factory.register_indexing_strategy("good", GoodStrategy)
    factory.register_indexing_strategy("weird", WeirdStrategy)
    
    # Good pipeline (no warnings, valid)
    indexing = factory.create_indexing_pipeline(["good"], [{}])
    retrieval = factory.create_retrieval_pipeline(["reranking"], [{}])
    
    validation = factory.validate_pipeline(indexing, retrieval)
    assert validation.is_valid
    
    # Weird pipeline (has warnings, but still valid if capabilities match)
    indexing = factory.create_indexing_pipeline(["weird"], [{}])
    
    # Console shows: ⚠️  WeirdStrategy: Produces VECTORS but doesn't require EMBEDDING
    
    validation = factory.validate_pipeline(indexing, retrieval)
    # Still valid if capabilities match (warnings don't block)
    assert validation.is_valid
    
    # Invalid pipeline (missing capabilities - this DOES block)
    indexing = factory.create_indexing_pipeline(["keyword_only"], [{}])
    validation = factory.validate_pipeline(indexing, retrieval)
    assert not validation.is_valid  # Fails validation
    assert IndexCapability.VECTORS in validation.missing_capabilities
```

**Story Points:** 13

---

## Sprint Planning

**Sprint 13:** Stories 12.1, 12.2, 12.3 (21 points)  
**Sprint 14:** Stories 12.4, 12.5 (21 points)  
**Sprint 15:** Story 12.6 (13 points)

---

## Capability Matrix

### What Capabilities Enable

| Capability | Enables | Example Strategies |
|------------|---------|-------------------|
| VECTORS | Semantic search, similarity matching | Vector search, reranking |
| KEYWORDS | BM25/keyword search | Keyword retrieval, hybrid search |
| GRAPH | Relationship traversal | Graph traversal, entity search |
| CHUNKS | Chunk-level retrieval | All retrieval strategies |
| HIERARCHY | Context expansion | Hierarchical retrieval |
| DATABASE | Persistence | All strategies (except in-memory) |

---

## Validation Examples

### Valid Combination

```python
# Indexing produces: {VECTORS, CHUNKS, DATABASE}
indexing = factory.create_indexing_pipeline(
    ["context_aware_chunking", "vector_embedding"],
    [{}, {}]
)

# Retrieval requires: {VECTORS}
retrieval = factory.create_retrieval_pipeline(
    ["query_expansion"],
    [{}]
)

# Validate
validation = factory.validate_pipeline(indexing, retrieval)
assert validation.is_valid  # ✅ VECTORS ⊆ {VECTORS, CHUNKS, DATABASE}
```

### Invalid Combination

```python
# Indexing produces: {KEYWORDS, DATABASE}
indexing = factory.create_indexing_pipeline(
    ["keyword_extraction"],
    [{}]
)

# Retrieval requires: {VECTORS}
retrieval = factory.create_retrieval_pipeline(
    ["reranking"],
    [{}]
)

# Validate
validation = factory.validate_pipeline(indexing, retrieval)
assert not validation.is_valid  # ❌ VECTORS ⊄ {KEYWORDS, DATABASE}
print(validation.message)
# Output: "Missing capabilities: ['VECTORS']"
print(validation.suggestions)
# Output: ["Add VectorEmbeddingIndexing to indexing pipeline"]
```

### Auto-Selection

```python
# Indexing produces many capabilities
indexing = factory.create_indexing_pipeline(
    ["context_aware_chunking", "vector_embedding", "keyword_extraction"],
    [{}, {}, {}]
)
# Capabilities: {CHUNKS, VECTORS, KEYWORDS, DATABASE}

# Auto-select compatible retrieval strategies
retrieval = factory.auto_select_retrieval(
    indexing,
    preferred_strategies=["reranking", "query_expansion", "keyword_retrieval"]
)
# Will include all three (all are compatible)
```

---

## Testing Strategy

### Unit Tests
- Capability enum operations
- IndexingResult creation and validation
- ValidationResult creation
- Interface contract tests

### Integration Tests
- Multi-strategy indexing pipelines
- Multi-strategy retrieval pipelines
- Capability validation
- Auto-selection logic

### End-to-End Tests
- Complete indexing → retrieval flow
- Invalid combination rejection
- Service + capability validation

---

## Documentation Updates

- [ ] Capability system guide
- [ ] Pipeline creation guide
- [ ] Validation and debugging guide
- [ ] Strategy compatibility matrix
- [ ] Migration guide (from old to new interfaces)
- [ ] Troubleshooting guide

---

## Success Criteria

- [ ] All capability enums defined
- [ ] Both interfaces (indexing + retrieval) implemented
- [ ] Both pipelines (indexing + retrieval) implemented
- [ ] Factory validation methods working
- [ ] Auto-selection working
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Migration path clear for existing strategies
- [ ] Invalid combinations caught at pipeline creation
- [ ] Helpful error messages with suggestions

---

## Benefits Achieved

**Separation of Concerns:**
- ✅ Indexing independent from retrieval
- ✅ Each has clear responsibilities
- ✅ Can evolve independently

**Flexibility:**
- ✅ Mix and match strategies freely
- ✅ Experiment with non-vector approaches
- ✅ Support radical RAG variants

**Safety:**
- ✅ Invalid combinations caught early
- ✅ Clear error messages
- ✅ Suggestions for fixes

**Maintainability:**
- ✅ Add new capabilities without breaking existing code
- ✅ Strategies self-describe requirements
- ✅ Factory handles complexity
