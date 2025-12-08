# Epic 11: Dependency Injection & Service Interface Decoupling

**Epic Goal:** Decouple all strategies from concrete service implementations by implementing dependency injection, allowing flexible service provider selection (ONNX for testing/CLI, API services for production) without code changes.

**Epic Story Points Total:** 39

**Dependencies:** Epic 10 (Lightweight Dependencies - COMPLETED ✅)

**Status:** Ready for implementation

---

## Background

Epic 10 successfully implemented ONNX-based services for lightweight local execution. However, the system currently lacks proper abstraction between strategies and service implementations. This epic establishes clear service interfaces and dependency injection to:

1. Allow strategies to work with any service implementation (ONNX, API-based, mock)
2. Enable testing without real API calls or model loading
3. Support mixed deployment scenarios (local embeddings + cloud LLM, etc.)
4. Prepare the foundation for the capability-based pipeline system (Epic 12)

---

## Service Interfaces to Define

### ILLMService
Used by: Query Expansion, Multi-Query, Agentic RAG, Self-Reflective RAG, Knowledge Graph, Contextual Retrieval

### IEmbeddingService
Used by: Vector Embedding, Context-Aware Chunking, Late Chunking, Fine-Tuned Embeddings

### IGraphService
Used by: Knowledge Graph Builder, Knowledge Graph Traversal

### IRerankingService
Used by: Re-ranking Strategy

### IDatabaseService
Used by: Most strategies (for persistence)

---

## Story 11.1: Define Service Interfaces

**As a** developer  
**I want** clear interfaces for all external services  
**So that** strategies depend on contracts, not implementations

**Acceptance Criteria:**
- Define `ILLMService` interface with `complete()` and `stream_complete()` methods
- Define `IEmbeddingService` interface with `embed()` and `embed_batch()` methods
- Define `IGraphService` interface with graph database operations
- Define `IRerankingService` interface with `rerank()` method
- Define `IDatabaseService` interface with repository operations
- Each interface is an Abstract Base Class with type hints
- Interface documentation with usage examples
- Unit tests demonstrating interface contracts

**Interface Specifications:**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

class ILLMService(ABC):
    """Interface for Large Language Model services"""
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate completion for prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated completion text
        """
        pass
    
    @abstractmethod
    async def stream_complete(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters
            
        Yields:
            Generated tokens as they arrive
        """
        pass

class IEmbeddingService(ABC):
    """Interface for embedding generation services"""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding vector dimension
        """
        pass

class IGraphService(ABC):
    """Interface for graph database services"""
    
    @abstractmethod
    async def create_node(
        self, 
        label: str, 
        properties: dict
    ) -> str:
        """
        Create a node in the graph.
        
        Args:
            label: Node label/type
            properties: Node properties
            
        Returns:
            Node ID
        """
        pass
    
    @abstractmethod
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: dict = None
    ):
        """
        Create relationship between nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Optional relationship properties
        """
        pass
    
    @abstractmethod
    async def query(self, cypher_query: str, parameters: dict = None) -> list[dict]:
        """
        Execute Cypher query.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        pass

class IRerankingService(ABC):
    """Interface for document reranking services"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: User query
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of (document_index, score) tuples
        """
        pass

class IDatabaseService(ABC):
    """Interface for database operations"""
    
    @abstractmethod
    async def store_chunks(self, chunks: list[dict]):
        """Store document chunks"""
        pass
    
    @abstractmethod
    async def search_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 10
    ) -> list[dict]:
        """Search chunks by similarity"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> dict:
        """Retrieve chunk by ID"""
        pass
```

**Story Points:** 8

---

## Story 11.2: Create StrategyDependencies Container

**As a** developer  
**I want** a container for strategy dependencies  
**So that** dependencies can be injected and validated

**Acceptance Criteria:**
- Create `StrategyDependencies` dataclass with optional service fields
- Implement `validate_for_strategy()` method
- Support dependency override per strategy
- Type hints for all service fields
- Unit tests for validation logic
- Documentation with usage examples

**Implementation:**

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

class ServiceDependency(Enum):
    """Services that strategies may depend on"""
    
    LLM = auto()
    EMBEDDING = auto()
    GRAPH = auto()
    DATABASE = auto()
    RERANKER = auto()

@dataclass
class StrategyDependencies:
    """Container for injected services"""
    
    llm_service: Optional[ILLMService] = None
    embedding_service: Optional[IEmbeddingService] = None
    graph_service: Optional[IGraphService] = None
    database_service: Optional[IDatabaseService] = None
    reranker_service: Optional[IRerankingService] = None
    
    def validate_for_strategy(
        self,
        required_services: set[ServiceDependency]
    ) -> tuple[bool, list[ServiceDependency]]:
        """
        Validate that all required services are present.
        
        Args:
            required_services: Set of required ServiceDependency enums
            
        Returns:
            Tuple of (is_valid, missing_services)
        """
        missing = []
        
        if ServiceDependency.LLM in required_services and not self.llm_service:
            missing.append(ServiceDependency.LLM)
        if ServiceDependency.EMBEDDING in required_services and not self.embedding_service:
            missing.append(ServiceDependency.EMBEDDING)
        if ServiceDependency.GRAPH in required_services and not self.graph_service:
            missing.append(ServiceDependency.GRAPH)
        if ServiceDependency.DATABASE in required_services and not self.database_service:
            missing.append(ServiceDependency.DATABASE)
        if ServiceDependency.RERANKER in required_services and not self.reranker_service:
            missing.append(ServiceDependency.RERANKER)
        
        return (len(missing) == 0, missing)
    
    def get_missing_services_message(
        self,
        required_services: set[ServiceDependency]
    ) -> str:
        """
        Get user-friendly error message for missing services.
        
        Args:
            required_services: Required services
            
        Returns:
            Error message or empty string if valid
        """
        is_valid, missing = self.validate_for_strategy(required_services)
        
        if is_valid:
            return ""
        
        service_names = [s.name for s in missing]
        return f"Missing required services: {', '.join(service_names)}"
```

**Story Points:** 5

---

## Story 11.3: Implement Service Implementations

**As a** developer  
**I want** concrete implementations of all service interfaces  
**So that** strategies can use real services

**Acceptance Criteria:**
- Implement `ONNXEmbeddingService` (already exists, needs interface conformance)
- Implement `ONNXLLMService` for local LLM execution
- Implement `AnthropicLLMService` for Claude API
- Implement `OpenAILLMService` for GPT API
- Implement `CohereRerankingService` for Cohere reranking API
- Implement `CosineRerankingService` for local similarity reranking
- Implement `Neo4jGraphService` for Neo4j database
- Implement `PostgresqlDatabaseService` for PostgreSQL with pgvector
- Each implementation passes interface tests
- Documentation for each service implementation

**ONNX Services (for testing/CLI POC):**
```python
class ONNXEmbeddingService(IEmbeddingService):
    """ONNX-based local embedding service"""
    
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        # Implementation from Epic 10
    
    async def embed(self, text: str) -> list[float]:
        # ONNX inference
        pass
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batch ONNX inference
        pass

class ONNXLLMService(ILLMService):
    """ONNX-based local LLM service (for testing)"""
    
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        # ONNX text generation
        pass
```

**API Services (for production):**
```python
class AnthropicLLMService(ILLMService):
    """Anthropic Claude API service"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI embedding API service"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
```

**Story Points:** 13

---

## Story 11.4: Update Strategy Base Classes for DI

**As a** developer  
**I want** strategy base classes to accept dependencies  
**So that** all strategies use dependency injection

**Acceptance Criteria:**
- Update strategy constructors to accept `dependencies: StrategyDependencies`
- Add `requires_services()` abstract method to base classes
- Validate dependencies at strategy instantiation
- Update all existing strategies to new constructor signature
- Backward compatibility during migration (optional)
- Unit tests for dependency validation
- Documentation with DI examples

**Updated Strategy Base:**

```python
class BaseStrategy(ABC):
    """Base class for all strategies"""
    
    def __init__(
        self, 
        config: dict, 
        dependencies: StrategyDependencies
    ):
        """
        Initialize strategy with configuration and dependencies.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services
            
        Raises:
            ValueError: If required services are missing
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        required = self.requires_services()
        is_valid, missing = dependencies.validate_for_strategy(required)
        
        if not is_valid:
            service_names = [s.name for s in missing]
            raise ValueError(
                f"{self.__class__.__name__} requires services: {', '.join(service_names)}"
            )
    
    @abstractmethod
    def requires_services(self) -> set[ServiceDependency]:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of required ServiceDependency enums
        """
        pass
```

**Example Strategy with DI:**

```python
class QueryExpansionStrategy(BaseStrategy):
    """Query expansion using LLM"""
    
    def requires_services(self) -> set[ServiceDependency]:
        return {ServiceDependency.LLM, ServiceDependency.DATABASE}
    
    async def expand_query(self, query: str) -> str:
        # Use injected LLM service
        prompt = f"Expand this query with relevant details: {query}"
        expanded = await self.deps.llm_service.complete(prompt)
        return expanded
```

**Story Points:** 5

---

## Story 11.5: Update RAGFactory for DI

**As a** developer  
**I want** the factory to inject dependencies into strategies  
**So that** service configuration is centralized

**Acceptance Criteria:**
- Update `RAGFactory` constructor to accept services
- Store services in `StrategyDependencies` container
- Inject dependencies when creating strategies
- Support per-strategy dependency override
- Validate dependencies before strategy creation
- Clear error messages for missing services
- Unit tests with various service configurations
- Documentation with factory usage examples

**Updated Factory:**

```python
class RAGFactory:
    """Factory for creating strategies with dependency injection"""
    
    def __init__(
        self,
        llm_service: Optional[ILLMService] = None,
        embedding_service: Optional[IEmbeddingService] = None,
        graph_service: Optional[IGraphService] = None,
        database_service: Optional[IDatabaseService] = None,
        reranker_service: Optional[IRerankingService] = None
    ):
        """
        Initialize factory with services.
        
        Args:
            llm_service: LLM service implementation
            embedding_service: Embedding service implementation
            graph_service: Graph database service implementation
            database_service: Database service implementation
            reranker_service: Reranking service implementation
        """
        self.dependencies = StrategyDependencies(
            llm_service=llm_service,
            embedding_service=embedding_service,
            graph_service=graph_service,
            database_service=database_service,
            reranker_service=reranker_service
        )
        self._strategy_registry = {}
    
    def create_strategy(
        self,
        strategy_name: str,
        config: dict,
        override_deps: Optional[StrategyDependencies] = None
    ):
        """
        Create strategy with dependency injection.
        
        Args:
            strategy_name: Name of strategy to create
            config: Strategy configuration
            override_deps: Optional dependency override
            
        Returns:
            Strategy instance with injected dependencies
            
        Raises:
            ValueError: If strategy not found or dependencies missing
        """
        if strategy_name not in self._strategy_registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = self._strategy_registry[strategy_name]
        deps = override_deps or self.dependencies
        
        # Strategy constructor will validate dependencies
        return strategy_class(config=config, dependencies=deps)
```

**Usage Examples:**

```python
# ONNX services for testing/CLI
from rag_factory.services import ONNXEmbeddingService, ONNXLLMService

embedding_svc = ONNXEmbeddingService(model_path="./models/embeddings.onnx")
llm_svc = ONNXLLMService(model_path="./models/llm.onnx")

factory = RAGFactory(
    embedding_service=embedding_svc,
    llm_service=llm_svc
)

# API services for production
from rag_factory.services import AnthropicLLMService, OpenAIEmbeddingService

llm_svc = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedding_svc = OpenAIEmbeddingService(api_key=os.getenv("OPENAI_API_KEY"))

factory = RAGFactory(
    llm_service=llm_svc,
    embedding_service=embedding_svc
)

# Mixed deployment (local embeddings, cloud LLM)
local_embeddings = ONNXEmbeddingService(model_path="./models/embeddings.onnx")
cloud_llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"))

factory = RAGFactory(
    embedding_service=local_embeddings,
    llm_service=cloud_llm
)

# Different LLMs for different strategies
cheap_llm = OpenAILLMService(model="gpt-3.5-turbo")
expensive_llm = AnthropicLLMService(model="claude-opus-4")

factory = RAGFactory(llm_service=cheap_llm)

# Override for specific strategy
kg_deps = StrategyDependencies(llm_service=expensive_llm, graph_service=neo4j_svc)
knowledge_graph = factory.create_strategy("knowledge_graph", {}, kg_deps)
```

**Story Points:** 3

---

## Story 11.6: Implement Consistency Checker

**As a** developer  
**I want** warnings when capabilities and services are inconsistent  
**So that** I can catch mistakes without blocking valid edge cases

**Acceptance Criteria:**
- Create `ConsistencyChecker` class with warning methods
- Define expected patterns (VECTORS → EMBEDDING, GRAPH → GRAPH service, etc.)
- Check indexing strategies for inconsistencies
- Check retrieval strategies for inconsistencies
- Return list of warning messages (don't raise exceptions)
- Integrate warnings into factory registration
- Integrate warnings into pipeline creation (optional)
- Unit tests with consistent and inconsistent strategies
- Documentation explaining warning system

**Warning Categories:**

1. **Indexing Inconsistencies:**
   - Produces VECTORS but doesn't require EMBEDDING
   - Produces GRAPH but doesn't require GRAPH service
   - Produces DATABASE capability but doesn't require DATABASE service
   - Declares IN_MEMORY but requires DATABASE service

2. **Retrieval Inconsistencies:**
   - Requires VECTORS but doesn't require DATABASE service
   - Requires GRAPH but doesn't require GRAPH service
   - Requires KEYWORDS but doesn't require DATABASE service

**Implementation:**

```python
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

class ConsistencyChecker:
    """Checks consistency between capabilities and services (warns, doesn't fail)"""
    
    # Common patterns - not exhaustive, just helpful hints
    EXPECTED_SERVICES_FOR_CAPABILITY = {
        IndexCapability.VECTORS: {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE},
        IndexCapability.KEYWORDS: {ServiceDependency.DATABASE},
        IndexCapability.GRAPH: {ServiceDependency.GRAPH, ServiceDependency.DATABASE},
        IndexCapability.CHUNKS: {ServiceDependency.DATABASE},
        IndexCapability.HIERARCHY: {ServiceDependency.DATABASE},
        IndexCapability.DATABASE: {ServiceDependency.DATABASE},
        # Note: IN_MEMORY and FILE_BACKED intentionally don't require DATABASE
    }
    
    def check_indexing_strategy(
        self,
        strategy: 'IIndexingStrategy'
    ) -> list[str]:
        """
        Check indexing strategy consistency, return warnings.
        
        Args:
            strategy: Strategy to check (can be temp instance)
            
        Returns:
            List of warning messages (empty if consistent)
        """
        warnings = []
        
        produces = strategy.produces()
        requires_services = strategy.requires_services()
        strategy_name = strategy.__class__.__name__
        
        # Check: Producing VECTORS without EMBEDDING service
        if IndexCapability.VECTORS in produces:
            if ServiceDependency.EMBEDDING not in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Produces VECTORS but doesn't require EMBEDDING service. "
                    f"This is unusual unless loading pre-computed embeddings."
                )
        
        # Check: Producing GRAPH without GRAPH service
        if IndexCapability.GRAPH in produces:
            if ServiceDependency.GRAPH not in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Produces GRAPH but doesn't require GRAPH service. "
                    f"This is unusual - graph strategies typically need graph database."
                )
        
        # Check: Producing DATABASE capability without DATABASE service
        if IndexCapability.DATABASE in produces:
            if ServiceDependency.DATABASE not in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Produces DATABASE capability but doesn't require DATABASE service. "
                    f"Did you mean to declare IN_MEMORY instead?"
                )
        
        # Check: Declaring IN_MEMORY but requiring DATABASE
        if IndexCapability.IN_MEMORY in produces:
            if ServiceDependency.DATABASE in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Declares IN_MEMORY capability but requires DATABASE service. "
                    f"This is unusual - in-memory strategies typically don't need database."
                )
        
        # Check: Producing KEYWORDS without DATABASE (might be intentional for in-memory)
        if IndexCapability.KEYWORDS in produces:
            if ServiceDependency.DATABASE not in requires_services:
                if IndexCapability.IN_MEMORY not in produces:
                    warnings.append(
                        f"⚠️  {strategy_name}: Produces KEYWORDS but doesn't require DATABASE service. "
                        f"Consider declaring IN_MEMORY if storing keywords in memory."
                    )
        
        return warnings
    
    def check_retrieval_strategy(
        self,
        strategy: 'IRetrievalStrategy'
    ) -> list[str]:
        """
        Check retrieval strategy consistency, return warnings.
        
        Args:
            strategy: Strategy to check (can be temp instance)
            
        Returns:
            List of warning messages (empty if consistent)
        """
        warnings = []
        
        requires_caps = strategy.requires()
        requires_services = strategy.requires_services()
        strategy_name = strategy.__class__.__name__
        
        # Check: Requiring VECTORS but not DATABASE service
        if IndexCapability.VECTORS in requires_caps:
            if ServiceDependency.DATABASE not in requires_services:
                if IndexCapability.IN_MEMORY not in requires_caps:
                    warnings.append(
                        f"⚠️  {strategy_name}: Requires VECTORS capability but doesn't require DATABASE service. "
                        f"This is unusual - vector search typically needs database access."
                    )
        
        # Check: Requiring GRAPH but not GRAPH service
        if IndexCapability.GRAPH in requires_caps:
            if ServiceDependency.GRAPH not in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Requires GRAPH capability but doesn't require GRAPH service. "
                    f"This is unusual - graph traversal typically needs graph database access."
                )
        
        # Check: Requiring KEYWORDS but not DATABASE
        if IndexCapability.KEYWORDS in requires_caps:
            if ServiceDependency.DATABASE not in requires_services:
                if IndexCapability.IN_MEMORY not in requires_caps:
                    warnings.append(
                        f"⚠️  {strategy_name}: Requires KEYWORDS capability but doesn't require DATABASE service. "
                        f"This is unusual - keyword search typically needs database access."
                    )
        
        # Check: Requiring HIERARCHY but not DATABASE
        if IndexCapability.HIERARCHY in requires_caps:
            if ServiceDependency.DATABASE not in requires_services:
                warnings.append(
                    f"⚠️  {strategy_name}: Requires HIERARCHY capability but doesn't require DATABASE service. "
                    f"This is unusual - hierarchical retrieval typically needs database access."
                )
        
        return warnings
    
    def check_and_log(
        self,
        strategy,
        strategy_type: str = "indexing"
    ):
        """
        Check strategy and log warnings.
        
        Args:
            strategy: Strategy instance to check
            strategy_type: "indexing" or "retrieval"
        """
        if strategy_type == "indexing":
            warnings = self.check_indexing_strategy(strategy)
        else:
            warnings = self.check_retrieval_strategy(strategy)
        
        for warning in warnings:
            logger.warning(warning)
        
        return len(warnings) == 0  # Returns True if consistent
```

**Integration into Factory:**

```python
class RAGFactory:
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.consistency_checker = ConsistencyChecker()
        self._indexing_registry = {}
        self._retrieval_registry = {}
    
    def register_indexing_strategy(
        self,
        name: str,
        strategy_class: type
    ):
        """
        Register indexing strategy and check consistency.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        # Create temp instance to check consistency
        temp = strategy_class.__new__(strategy_class)
        
        # Check consistency (warns but doesn't fail)
        warnings = self.consistency_checker.check_indexing_strategy(temp)
        for warning in warnings:
            logger.warning(warning)
        
        # Register regardless of warnings
        self._indexing_registry[name] = strategy_class
    
    def register_retrieval_strategy(
        self,
        name: str,
        strategy_class: type
    ):
        """Register retrieval strategy and check consistency"""
        temp = strategy_class.__new__(strategy_class)
        
        warnings = self.consistency_checker.check_retrieval_strategy(temp)
        for warning in warnings:
            logger.warning(warning)
        
        self._retrieval_registry[name] = strategy_class
    
    def check_all_strategies(self) -> dict[str, list[str]]:
        """
        Check all registered strategies for consistency.
        
        Returns:
            Dict mapping strategy names to their warnings
        """
        results = {}
        
        # Check indexing strategies
        for name, strategy_class in self._indexing_registry.items():
            temp = strategy_class.__new__(strategy_class)
            warnings = self.consistency_checker.check_indexing_strategy(temp)
            if warnings:
                results[f"indexing:{name}"] = warnings
        
        # Check retrieval strategies
        for name, strategy_class in self._retrieval_registry.items():
            temp = strategy_class.__new__(strategy_class)
            warnings = self.consistency_checker.check_retrieval_strategy(temp)
            if warnings:
                results[f"retrieval:{name}"] = warnings
        
        return results
```

**Usage Examples:**

```python
# Automatic checking during registration
factory = RAGFactory(dependencies=deps)
factory.register_indexing_strategy("my_strategy", MyStrategy)
# Logs warnings automatically if inconsistent

# Manual checking of all strategies
warnings = factory.check_all_strategies()
if warnings:
    print("Consistency warnings found:")
    for strategy_name, msgs in warnings.items():
        print(f"\n{strategy_name}:")
        for msg in msgs:
            print(f"  {msg}")
else:
    print("✅ All strategies consistent")

# Example output:
# indexing:weird_strategy:
#   ⚠️  WeirdStrategy: Produces VECTORS but doesn't require EMBEDDING service.
#      This is unusual unless loading pre-computed embeddings.
```

**CLI Integration (for Story 8.5.1):**

```bash
# Check all registered strategies
$ rag-factory check-consistency

Checking registered strategies for consistency...

Indexing Strategies:
  ✅ context_aware_chunking
  ✅ vector_embedding
  ✅ keyword_extraction
  ⚠️  experimental_strategy
     → Produces VECTORS but doesn't require EMBEDDING service

Retrieval Strategies:
  ✅ reranking
  ✅ query_expansion
  ✅ hierarchical_rag

Summary: 1 warning found (5 strategies checked)
```

**Testing Example:**

```python
def test_consistency_checker():
    checker = ConsistencyChecker()
    
    # Consistent strategy (no warnings)
    class GoodStrategy(IIndexingStrategy):
        def produces(self):
            return {IndexCapability.VECTORS, IndexCapability.DATABASE}
        
        def requires_services(self):
            return {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
    
    good = GoodStrategy()
    warnings = checker.check_indexing_strategy(good)
    assert len(warnings) == 0  # ✅ Consistent
    
    # Inconsistent strategy (has warnings)
    class WeirdStrategy(IIndexingStrategy):
        def produces(self):
            return {IndexCapability.VECTORS, IndexCapability.DATABASE}
        
        def requires_services(self):
            return {ServiceDependency.DATABASE}  # Missing EMBEDDING!
    
    weird = WeirdStrategy()
    warnings = checker.check_indexing_strategy(weird)
    assert len(warnings) == 1  # ⚠️  Has warning
    assert "EMBEDDING" in warnings[0]
    
    # Strategy still works despite warning
    # (warnings don't prevent instantiation)
```

**Documentation Points:**

- Warnings are advisory, not blocking
- Common patterns are checked, but edge cases are allowed
- Developers can ignore warnings for intentional deviations
- Warnings help catch copy-paste errors and typos
- Consistency checker can be extended with new patterns

**Story Points:** 5

---

## Sprint Planning

**Sprint 11:** Stories 11.1, 11.2, 11.4 (18 points)  
**Sprint 12:** Stories 11.3, 11.5, 11.6 (21 points)

---

## Service Strategy Matrix

### Strategies and Required Services

| Strategy | LLM | EMBEDDING | GRAPH | DATABASE | RERANKER |
|----------|-----|-----------|-------|----------|----------|
| Query Expansion | ✅ | ❌ | ❌ | ✅ | ❌ |
| Multi-Query RAG | ✅ | ❌ | ❌ | ✅ | ❌ |
| Agentic RAG | ✅ | ❌ | ❌ | ✅ | ❌ |
| Self-Reflective RAG | ✅ | ❌ | ❌ | ✅ | ❌ |
| Contextual Retrieval | ✅ | ❌ | ❌ | ✅ | ❌ |
| Knowledge Graph | ✅ | ❌ | ✅ | ✅ | ❌ |
| Context-Aware Chunking | ❌ | ✅ | ❌ | ✅ | ❌ |
| Vector Embedding | ❌ | ✅ | ❌ | ✅ | ❌ |
| Late Chunking | ❌ | ✅ | ❌ | ✅ | ❌ |
| Fine-Tuned Embeddings | ❌ | ✅ | ❌ | ✅ | ❌ |
| Re-ranking | ❌ | ❌ | ❌ | ✅ | ✅ |
| Hierarchical RAG | ❌ | ❌ | ❌ | ✅ | ❌ |
| Keyword Retrieval | ❌ | ❌ | ❌ | ✅ | ❌ |

---

## Testing Strategy

### Unit Tests
- Interface contract tests
- Dependency validation tests
- Mock service implementations
- Strategy instantiation with/without dependencies

### Integration Tests
- Real ONNX services
- Real API services (optional, with API keys)
- Mixed service configurations
- Error handling for missing services

### Test Fixtures
```python
@pytest.fixture
def mock_llm_service():
    class MockLLM(ILLMService):
        async def complete(self, prompt, **kwargs):
            return f"Mock response to: {prompt}"
        
        async def stream_complete(self, prompt, **kwargs):
            yield "Mock"
            yield " streaming"
    
    return MockLLM()

@pytest.fixture
def onnx_embedding_service():
    return ONNXEmbeddingService(model_path="./tests/fixtures/model.onnx")
```

---

## Documentation Updates

- [ ] Service interface documentation
- [ ] Dependency injection guide
- [ ] Service implementation guide (for custom services)
- [ ] Testing with mocks guide
- [ ] Production deployment guide (API services)
- [ ] CLI/POC guide (ONNX services)
- [ ] Migration guide (for existing code)

---

## Success Criteria

- [ ] All service interfaces defined and documented
- [ ] StrategyDependencies container implemented
- [ ] All service implementations created
- [ ] All strategies use dependency injection
- [ ] Factory injects dependencies correctly
- [ ] Dependency validation works
- [ ] Unit tests pass with mocks
- [ ] Integration tests pass with ONNX services
- [ ] Documentation complete
- [ ] Zero import-time dependencies on concrete services
- [ ] Strategies work with any service implementation

---

## Benefits Achieved

**Flexibility:**
- ✅ Use ONNX for testing/CLI, APIs for production
- ✅ Mix local and cloud services
- ✅ Different LLMs for different strategies

**Testability:**
- ✅ Mock all external services
- ✅ Unit tests without models or APIs
- ✅ Fast test execution

**Maintainability:**
- ✅ Clear separation of concerns
- ✅ Easy to add new service implementations
- ✅ Strategies don't depend on service details

**Deployment:**
- ✅ Choose optimal services per environment
- ✅ Cost optimization (cheap vs expensive services)
- ✅ Works offline (ONNX) or online (APIs)
