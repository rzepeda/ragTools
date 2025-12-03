# Story 4.3: Implement Query Expansion Strategy

**Story ID:** 4.3
**Epic:** Epic 4 - Priority RAG Strategies
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 3 (LLM Service)

---

## User Story

**As a** system
**I want** to expand user queries with more specific details
**So that** search precision and recall improve

---

## Detailed Requirements

### Functional Requirements

1. **LLM-Based Query Expansion**
   - Use LLM to analyze and expand user queries
   - Generate additional search terms and synonyms
   - Add context and specificity to vague queries
   - Preserve original query intent
   - Support multiple expansion strategies (keywords, questions, reformulation)

2. **Multiple Expansion Techniques**
   - **Keyword Expansion**: Add relevant keywords and synonyms
   - **Query Reformulation**: Rephrase query for better search
   - **Question Generation**: Generate related questions
   - **Multi-Query**: Generate multiple variations of the query
   - **Hypothetical Document Expansion (HyDE)**: Generate hypothetical answer and search for it

3. **Configurable Expansion Instructions**
   - Customizable system prompts for expansion
   - Domain-specific expansion rules
   - Control expansion verbosity (1-5 additional terms vs full paragraphs)
   - Strategy-specific configurations

4. **Return Original and Expanded Queries**
   - Always preserve original query
   - Return expanded query with annotations
   - Track which parts were added by expansion
   - Support multiple expanded variants

5. **Search with Expanded Query**
   - Use expanded query for retrieval
   - Optionally combine original and expanded results
   - Merge and deduplicate results
   - Weight original vs expanded results

6. **Logging and Debugging**
   - Log all query expansions
   - Track expansion quality metrics
   - Record LLM responses
   - Enable/disable expansion for A/B testing
   - Performance timing metrics

7. **A/B Testing Capability**
   - Support for enabling/disabling expansion per request
   - Track metrics for expanded vs non-expanded queries
   - Compare search quality (precision, recall, NDCG)
   - Statistical significance testing

### Non-Functional Requirements

1. **Performance**
   - Query expansion in <1 second
   - Support concurrent expansion requests
   - Cache expanded queries
   - Batch processing capability

2. **Quality**
   - Expanded queries maintain original intent
   - Measurable improvement in search results
   - No hallucinated or irrelevant terms
   - Consistent expansion quality

3. **Reliability**
   - Handle LLM failures gracefully
   - Fallback to original query on errors
   - Timeout handling (max 5 seconds)
   - Retry logic for transient failures

4. **Observability**
   - Log all expansions with reasoning
   - Track expansion success rates
   - Monitor LLM costs
   - A/B test results visible

5. **Flexibility**
   - Support multiple LLM providers
   - Easy to customize expansion prompts
   - Configurable expansion strategies
   - Domain-specific adaptations

---

## Acceptance Criteria

### AC1: LLM-Based Expansion
- [ ] LLM integration for query expansion working
- [ ] Expansion preserves original query intent
- [ ] Multiple expansion strategies implemented
- [ ] Configurable system prompts
- [ ] Domain-specific expansion support

### AC2: Expansion Techniques
- [ ] Keyword expansion implemented
- [ ] Query reformulation implemented
- [ ] Question generation implemented
- [ ] Multi-query generation implemented
- [ ] HyDE (Hypothetical Document Expansion) implemented
- [ ] Strategy selection via configuration

### AC3: Configuration System
- [ ] Customizable expansion prompts
- [ ] Expansion verbosity control
- [ ] Strategy-specific settings
- [ ] Domain rules configurable
- [ ] Enable/disable per request

### AC4: Query Tracking
- [ ] Original query preserved
- [ ] Expanded query returned
- [ ] Expansion annotations tracked
- [ ] Multiple variants supported
- [ ] Reasoning logged

### AC5: Search Integration
- [ ] Expanded query used for retrieval
- [ ] Original + expanded results combined
- [ ] Deduplication working
- [ ] Result weighting configurable
- [ ] Merged results ranked correctly

### AC6: Logging and Debugging
- [ ] All expansions logged
- [ ] Quality metrics tracked
- [ ] LLM responses recorded
- [ ] Performance timing measured
- [ ] Debug mode available

### AC7: A/B Testing
- [ ] Enable/disable expansion per request
- [ ] Metrics tracked for both modes
- [ ] Search quality comparison
- [ ] Statistical testing implemented
- [ ] Results dashboard available

### AC8: Testing
- [ ] Unit tests for all expansion strategies (>90% coverage)
- [ ] Integration tests with real LLM
- [ ] Performance benchmarks meet <1s requirement
- [ ] Quality tests validate improvements
- [ ] A/B testing framework validated

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── query_expansion/
│   │   ├── __init__.py
│   │   ├── base.py                    # Base query expander interface
│   │   ├── expander_service.py        # Main expansion service
│   │   ├── llm_expander.py            # LLM-based expansion
│   │   ├── keyword_expander.py        # Keyword expansion
│   │   ├── multi_query_expander.py    # Multi-query generation
│   │   ├── hyde_expander.py           # HyDE implementation
│   │   ├── prompts.py                 # Expansion prompts
│   │   ├── cache.py                   # Expansion cache
│   │   ├── config.py                  # Configuration
│   │   └── metrics.py                 # Quality metrics
│
tests/
├── unit/
│   └── strategies/
│       └── query_expansion/
│           ├── test_expander_service.py
│           ├── test_llm_expander.py
│           ├── test_keyword_expander.py
│           ├── test_multi_query_expander.py
│           ├── test_hyde_expander.py
│           └── test_metrics.py
│
├── integration/
│   └── strategies/
│       └── test_query_expansion_integration.py
```

### Dependencies
```python
# requirements.txt additions
# LLM integration already in Epic 3
nltk==3.8.1                # For keyword extraction
spacy==3.7.0               # NLP for query analysis
scikit-learn==1.3.0        # For similarity metrics
```

### Base Query Expander Interface
```python
# rag_factory/strategies/query_expansion/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class ExpansionStrategy(Enum):
    """Enumeration of query expansion strategies."""
    KEYWORD = "keyword"
    REFORMULATION = "reformulation"
    QUESTION_GENERATION = "question_generation"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"  # Hypothetical Document Expansion

@dataclass
class ExpandedQuery:
    """Result from query expansion."""
    original_query: str
    expanded_query: str
    expansion_strategy: ExpansionStrategy
    added_terms: List[str]
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpansionResult:
    """Complete result from query expansion service."""
    original_query: str
    expanded_queries: List[ExpandedQuery]
    primary_expansion: ExpandedQuery  # Main expanded query to use
    execution_time_ms: float
    cache_hit: bool
    llm_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""
    strategy: ExpansionStrategy = ExpansionStrategy.KEYWORD

    # LLM settings
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.3  # Lower for more consistent expansions

    # Expansion settings
    max_additional_terms: int = 5
    generate_multiple_variants: bool = False
    num_variants: int = 3

    # Strategy-specific settings
    include_synonyms: bool = True
    include_related_terms: bool = True
    preserve_query_structure: bool = True

    # Prompt customization
    system_prompt: Optional[str] = None
    domain_context: Optional[str] = None

    # Performance settings
    enable_cache: bool = True
    cache_ttl: int = 3600
    timeout_seconds: float = 5.0

    # A/B testing
    enable_expansion: bool = True
    track_metrics: bool = True

    # Additional config
    extra_config: Dict[str, Any] = field(default_factory=dict)

class IQueryExpander(ABC):
    """Abstract base class for query expansion strategies."""

    def __init__(self, config: ExpansionConfig):
        """Initialize expander with configuration."""
        self.config = config

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query to improve search.

        Args:
            query: The original user query

        Returns:
            ExpandedQuery with expansion details
        """
        pass

    def validate_query(self, query: str) -> None:
        """Validate input query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 1000:
            raise ValueError(f"Query too long: {len(query)} characters (max: 1000)")

    def extract_added_terms(self, original: str, expanded: str) -> List[str]:
        """Extract terms that were added during expansion."""
        original_terms = set(original.lower().split())
        expanded_terms = set(expanded.lower().split())
        added = expanded_terms - original_terms
        return list(added)
```

### LLM-Based Expander Implementation
```python
# rag_factory/strategies/query_expansion/llm_expander.py
from typing import Optional, List
from .base import IQueryExpander, ExpandedQuery, ExpansionConfig, ExpansionStrategy
from .prompts import ExpansionPrompts
from ...services.llm.service import LLMService

class LLMQueryExpander(IQueryExpander):
    """
    Query expander using LLM to intelligently expand queries.
    Supports multiple expansion strategies via different prompts.
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service
        self.prompts = ExpansionPrompts(config)

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query using LLM."""
        self.validate_query(query)

        # Get appropriate prompt for strategy
        system_prompt = self.prompts.get_system_prompt(self.config.strategy)
        user_prompt = self.prompts.get_user_prompt(query, self.config.strategy)

        # Add domain context if provided
        if self.config.domain_context:
            system_prompt = f"{system_prompt}\n\nDomain context: {self.config.domain_context}"

        # Call LLM
        response = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        expanded_query = response.text.strip()

        # Extract added terms
        added_terms = self.extract_added_terms(query, expanded_query)

        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            expansion_strategy=self.config.strategy,
            added_terms=added_terms,
            reasoning=self._extract_reasoning(response.text),
            confidence=1.0,
            metadata={
                "llm_model": self.config.llm_model,
                "tokens_used": response.usage.get("total_tokens", 0)
            }
        )

    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from LLM response if present."""
        # Some prompts ask LLM to explain reasoning
        # This could be parsed from structured output
        return None


# rag_factory/strategies/query_expansion/prompts.py
from typing import Dict
from .base import ExpansionStrategy, ExpansionConfig

class ExpansionPrompts:
    """Prompt templates for different expansion strategies."""

    def __init__(self, config: ExpansionConfig):
        self.config = config

    def get_system_prompt(self, strategy: ExpansionStrategy) -> str:
        """Get system prompt for expansion strategy."""

        # Allow custom system prompt override
        if self.config.system_prompt:
            return self.config.system_prompt

        prompts = {
            ExpansionStrategy.KEYWORD: """You are a search query optimization expert.
Your task is to expand user queries by adding relevant keywords and synonyms to improve search results.
Add specific terms that capture the query's intent without changing its meaning.
Keep expansions concise and focused.""",

            ExpansionStrategy.REFORMULATION: """You are a search query optimization expert.
Your task is to reformulate user queries to make them more specific and searchable.
Rephrase the query to be clearer and more precise while preserving the original intent.
Focus on making the query more actionable for retrieval.""",

            ExpansionStrategy.QUESTION_GENERATION: """You are a search query optimization expert.
Your task is to convert user queries into well-formed questions that capture their information need.
Generate clear, specific questions that would help find relevant information.""",

            ExpansionStrategy.MULTI_QUERY: """You are a search query optimization expert.
Your task is to generate multiple variations of the user's query to improve search coverage.
Create diverse queries that capture different aspects of the user's information need.""",

            ExpansionStrategy.HYDE: """You are a search query optimization expert.
Your task is to generate a hypothetical document or passage that would answer the user's query.
Create a realistic, detailed response that contains the information the user is looking for.
This will be used to search for similar real documents."""
        }

        return prompts.get(strategy, prompts[ExpansionStrategy.KEYWORD])

    def get_user_prompt(self, query: str, strategy: ExpansionStrategy) -> str:
        """Get user prompt with the query for expansion."""

        prompts = {
            ExpansionStrategy.KEYWORD: f"""Original query: "{query}"

Expand this query by adding {self.config.max_additional_terms} relevant keywords or synonyms.
Return only the expanded query, nothing else.

Expanded query:""",

            ExpansionStrategy.REFORMULATION: f"""Original query: "{query}"

Reformulate this query to be more specific and searchable.
Return only the reformulated query, nothing else.

Reformulated query:""",

            ExpansionStrategy.QUESTION_GENERATION: f"""Original query: "{query}"

Convert this into a clear, specific question.
Return only the question, nothing else.

Question:""",

            ExpansionStrategy.MULTI_QUERY: f"""Original query: "{query}"

Generate {self.config.num_variants} different variations of this query.
Each variation should capture a different aspect or perspective.
Return only the queries, one per line.

Variations:""",

            ExpansionStrategy.HYDE: f"""Original query: "{query}"

Generate a hypothetical document passage (2-3 sentences) that would perfectly answer this query.
Be specific and detailed.
Return only the passage, nothing else.

Passage:"""
        }

        return prompts.get(strategy, prompts[ExpansionStrategy.KEYWORD])
```

### Expander Service
```python
# rag_factory/strategies/query_expansion/expander_service.py
from typing import List, Dict, Any, Optional
import time
import logging
from .base import IQueryExpander, ExpansionConfig, ExpansionResult, ExpandedQuery, ExpansionStrategy
from .llm_expander import LLMQueryExpander
from .cache import ExpansionCache
from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)

class QueryExpanderService:
    """
    Service for expanding user queries to improve search.

    Example:
        config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            max_additional_terms=5
        )
        service = QueryExpanderService(config, llm_service)
        result = service.expand("machine learning")
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        self.config = config
        self.llm_service = llm_service
        self.expander = LLMQueryExpander(config, llm_service)
        self.cache = ExpansionCache(config) if config.enable_cache else None
        self._stats = {
            "total_expansions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time_ms": 0.0,
            "expansion_enabled_count": 0,
            "expansion_disabled_count": 0
        }

    def expand(
        self,
        query: str,
        enable_expansion: Optional[bool] = None
    ) -> ExpansionResult:
        """
        Expand a query to improve search.

        Args:
            query: The original user query
            enable_expansion: Override config to enable/disable expansion (for A/B testing)

        Returns:
            ExpansionResult with original and expanded queries
        """
        start_time = time.time()
        self._stats["total_expansions"] += 1

        # Check if expansion is enabled
        expansion_enabled = enable_expansion if enable_expansion is not None else self.config.enable_expansion

        if expansion_enabled:
            self._stats["expansion_enabled_count"] += 1
        else:
            self._stats["expansion_disabled_count"] += 1

        # If disabled, return original query
        if not expansion_enabled:
            return self._create_passthrough_result(query, start_time)

        # Check cache
        cache_hit = False
        if self.cache:
            cache_key = self._compute_cache_key(query)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self._stats["cache_hits"] += 1
                cached_result.cache_hit = True
                return cached_result

        self._stats["cache_misses"] += 1

        try:
            # Expand query
            if self.config.generate_multiple_variants:
                expanded_queries = self._expand_multiple(query)
            else:
                expanded_query = self.expander.expand(query)
                expanded_queries = [expanded_query]

            # Select primary expansion
            primary_expansion = expanded_queries[0]

            execution_time_ms = (time.time() - start_time) * 1000

            result = ExpansionResult(
                original_query=query,
                expanded_queries=expanded_queries,
                primary_expansion=primary_expansion,
                execution_time_ms=execution_time_ms,
                cache_hit=cache_hit,
                llm_used=self.config.llm_model
            )

            # Cache the result
            if self.cache:
                self.cache.set(cache_key, result)

            # Update stats
            self._update_avg_execution_time(execution_time_ms)

            # Log expansion
            if self.config.track_metrics:
                self._log_expansion(result)

            return result

        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            # Fallback to original query
            return self._create_passthrough_result(query, start_time, error=str(e))

    def _expand_multiple(self, query: str) -> List[ExpandedQuery]:
        """Generate multiple query variations."""
        expansions = []

        if self.config.strategy == ExpansionStrategy.MULTI_QUERY:
            # LLM will generate multiple queries
            expanded = self.expander.expand(query)

            # Parse multiple queries from response
            queries = expanded.expanded_query.split('\n')
            for q in queries[:self.config.num_variants]:
                if q.strip():
                    expansions.append(ExpandedQuery(
                        original_query=query,
                        expanded_query=q.strip(),
                        expansion_strategy=self.config.strategy,
                        added_terms=self.expander.extract_added_terms(query, q.strip()),
                        confidence=1.0
                    ))
        else:
            # Generate single expansion
            expansions.append(self.expander.expand(query))

        return expansions if expansions else [ExpandedQuery(
            original_query=query,
            expanded_query=query,
            expansion_strategy=self.config.strategy,
            added_terms=[],
            confidence=1.0
        )]

    def _create_passthrough_result(
        self,
        query: str,
        start_time: float,
        error: Optional[str] = None
    ) -> ExpansionResult:
        """Create result that passes through original query."""
        execution_time_ms = (time.time() - start_time) * 1000

        passthrough = ExpandedQuery(
            original_query=query,
            expanded_query=query,
            expansion_strategy=self.config.strategy,
            added_terms=[],
            confidence=1.0,
            metadata={"passthrough": True, "error": error} if error else {"passthrough": True}
        )

        return ExpansionResult(
            original_query=query,
            expanded_queries=[passthrough],
            primary_expansion=passthrough,
            execution_time_ms=execution_time_ms,
            cache_hit=False,
            metadata={"expansion_disabled": True} if not error else {"error": error}
        )

    def _compute_cache_key(self, query: str) -> str:
        """Compute cache key for query."""
        import hashlib

        content = f"{query}:{self.config.strategy.value}:{self.config.llm_model}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _update_avg_execution_time(self, new_time_ms: float):
        """Update average execution time."""
        total = self._stats["total_expansions"]
        current_avg = self._stats["avg_execution_time_ms"]

        new_avg = ((current_avg * (total - 1)) + new_time_ms) / total
        self._stats["avg_execution_time_ms"] = new_avg

    def _log_expansion(self, result: ExpansionResult):
        """Log expansion for analysis."""
        logger.info(
            f"Query expanded: '{result.original_query}' -> '{result.primary_expansion.expanded_query}' "
            f"(+{len(result.primary_expansion.added_terms)} terms, {result.execution_time_ms:.0f}ms)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_hit_rate = 0.0
        if self._stats["total_expansions"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / self._stats["total_expansions"]

        expansion_rate = 0.0
        if self._stats["total_expansions"] > 0:
            expansion_rate = self._stats["expansion_enabled_count"] / self._stats["total_expansions"]

        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "expansion_rate": expansion_rate,
            "strategy": self.config.strategy.value,
            "llm_model": self.config.llm_model
        }

    def clear_cache(self):
        """Clear the expansion cache."""
        if self.cache:
            self.cache.clear()
```

### HyDE Implementation
```python
# rag_factory/strategies/query_expansion/hyde_expander.py
from .base import IQueryExpander, ExpandedQuery, ExpansionConfig, ExpansionStrategy
from .prompts import ExpansionPrompts
from ...services.llm.service import LLMService

class HyDEExpander(IQueryExpander):
    """
    Hypothetical Document Expansion (HyDE).

    Instead of expanding the query with keywords, HyDE generates a hypothetical
    document/passage that would answer the query, then uses that for retrieval.
    This often works better than keyword expansion for semantic search.
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service
        self.prompts = ExpansionPrompts(config)

    def expand(self, query: str) -> ExpandedQuery:
        """Generate hypothetical document for query."""
        self.validate_query(query)

        system_prompt = self.prompts.get_system_prompt(ExpansionStrategy.HYDE)
        user_prompt = self.prompts.get_user_prompt(query, ExpansionStrategy.HYDE)

        # Generate hypothetical document
        response = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=0.7  # Slightly higher for more creative responses
        )

        hypothetical_doc = response.text.strip()

        return ExpandedQuery(
            original_query=query,
            expanded_query=hypothetical_doc,
            expansion_strategy=ExpansionStrategy.HYDE,
            added_terms=[],  # Not applicable for HyDE
            reasoning="Generated hypothetical document for semantic search",
            confidence=1.0,
            metadata={
                "method": "HyDE",
                "tokens_used": response.usage.get("total_tokens", 0)
            }
        )
```

---

## Unit Tests

### Test File Location
`tests/unit/strategies/query_expansion/test_expander_service.py`
`tests/unit/strategies/query_expansion/test_llm_expander.py`

### Test Cases

#### TC4.3.1: Expander Service Tests
```python
import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.query_expansion.expander_service import QueryExpanderService
from rag_factory.strategies.query_expansion.base import ExpansionConfig, ExpansionStrategy, ExpandedQuery

@pytest.fixture
def expansion_config():
    return ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        max_additional_terms=5,
        enable_cache=True
    )

@pytest.fixture
def mock_llm_service():
    service = Mock()
    service.generate.return_value = Mock(
        text="machine learning algorithms neural networks deep learning",
        usage={"total_tokens": 20}
    )
    return service

def test_service_initialization(expansion_config, mock_llm_service):
    """Test service initializes correctly."""
    service = QueryExpanderService(expansion_config, mock_llm_service)

    assert service.config == expansion_config
    assert service.llm_service == mock_llm_service

def test_expand_basic(expansion_config, mock_llm_service):
    """Test basic query expansion."""
    service = QueryExpanderService(expansion_config, mock_llm_service)

    result = service.expand("machine learning")

    assert result.original_query == "machine learning"
    assert result.primary_expansion.expanded_query != ""
    assert len(result.expanded_queries) >= 1
    assert result.cache_hit == False

def test_expand_with_cache_hit(expansion_config, mock_llm_service):
    """Test expansion with cache hit."""
    service = QueryExpanderService(expansion_config, mock_llm_service)

    # First call - cache miss
    result1 = service.expand("test query")
    assert result1.cache_hit == False

    # Second call - cache hit
    result2 = service.expand("test query")
    assert result2.cache_hit == True

    # LLM should only be called once
    assert mock_llm_service.generate.call_count == 1

def test_expand_disabled(expansion_config, mock_llm_service):
    """Test expansion when disabled."""
    service = QueryExpanderService(expansion_config, mock_llm_service)

    result = service.expand("test query", enable_expansion=False)

    # Should return original query
    assert result.original_query == "test query"
    assert result.primary_expansion.expanded_query == "test query"
    assert len(result.primary_expansion.added_terms) == 0

    # LLM should not be called
    assert mock_llm_service.generate.call_count == 0

def test_expand_with_error_fallback(expansion_config, mock_llm_service):
    """Test fallback to original query on error."""
    mock_llm_service.generate.side_effect = Exception("LLM error")

    service = QueryExpanderService(expansion_config, mock_llm_service)

    result = service.expand("test query")

    # Should fallback to original query
    assert result.original_query == "test query"
    assert result.primary_expansion.expanded_query == "test query"
    assert "error" in result.metadata

def test_multiple_variants(mock_llm_service):
    """Test generating multiple query variants."""
    mock_llm_service.generate.return_value = Mock(
        text="query variant 1\nquery variant 2\nquery variant 3",
        usage={"total_tokens": 30}
    )

    config = ExpansionConfig(
        strategy=ExpansionStrategy.MULTI_QUERY,
        generate_multiple_variants=True,
        num_variants=3
    )

    service = QueryExpanderService(config, mock_llm_service)

    result = service.expand("original query")

    assert len(result.expanded_queries) >= 1
    # Should have multiple variants
    if result.expanded_queries[0].expansion_strategy == ExpansionStrategy.MULTI_QUERY:
        assert len(result.expanded_queries) > 1

def test_get_stats(expansion_config, mock_llm_service):
    """Test statistics tracking."""
    service = QueryExpanderService(expansion_config, mock_llm_service)

    service.expand("query 1")
    service.expand("query 1")  # Cache hit
    service.expand("query 2")
    service.expand("query 3", enable_expansion=False)

    stats = service.get_stats()

    assert stats["total_expansions"] == 4
    assert stats["cache_hits"] >= 1
    assert stats["expansion_enabled_count"] == 3
    assert stats["expansion_disabled_count"] == 1
    assert "cache_hit_rate" in stats
    assert "expansion_rate" in stats
```

#### TC4.3.2: LLM Expander Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.strategies.query_expansion.llm_expander import LLMQueryExpander
from rag_factory.strategies.query_expansion.base import ExpansionConfig, ExpansionStrategy

@pytest.fixture
def expansion_config():
    return ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        max_additional_terms=5
    )

@pytest.fixture
def mock_llm_service():
    service = Mock()
    service.generate.return_value = Mock(
        text="machine learning algorithms neural networks",
        usage={"total_tokens": 15}
    )
    return service

def test_llm_expander_initialization(expansion_config, mock_llm_service):
    """Test LLM expander initializes correctly."""
    expander = LLMQueryExpander(expansion_config, mock_llm_service)

    assert expander.config == expansion_config
    assert expander.llm_service == mock_llm_service

def test_expand_query(expansion_config, mock_llm_service):
    """Test query expansion."""
    expander = LLMQueryExpander(expansion_config, mock_llm_service)

    result = expander.expand("machine learning")

    assert result.original_query == "machine learning"
    assert result.expanded_query != ""
    assert result.expansion_strategy == ExpansionStrategy.KEYWORD
    assert isinstance(result.added_terms, list)

def test_validate_query_empty(expansion_config, mock_llm_service):
    """Test validation rejects empty query."""
    expander = LLMQueryExpander(expansion_config, mock_llm_service)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        expander.validate_query("")

def test_validate_query_too_long(expansion_config, mock_llm_service):
    """Test validation rejects overly long query."""
    expander = LLMQueryExpander(expansion_config, mock_llm_service)

    long_query = "x" * 1001
    with pytest.raises(ValueError, match="Query too long"):
        expander.validate_query(long_query)

def test_extract_added_terms(expansion_config, mock_llm_service):
    """Test extraction of added terms."""
    expander = LLMQueryExpander(expansion_config, mock_llm_service)

    original = "machine learning"
    expanded = "machine learning algorithms neural networks deep learning"

    added = expander.extract_added_terms(original, expanded)

    assert "algorithms" in added
    assert "neural" in added
    assert "networks" in added

def test_different_expansion_strategies(mock_llm_service):
    """Test different expansion strategies use different prompts."""
    strategies = [
        ExpansionStrategy.KEYWORD,
        ExpansionStrategy.REFORMULATION,
        ExpansionStrategy.QUESTION_GENERATION
    ]

    for strategy in strategies:
        config = ExpansionConfig(strategy=strategy)
        expander = LLMQueryExpander(config, mock_llm_service)

        result = expander.expand("test query")

        assert result.expansion_strategy == strategy
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_query_expansion_integration.py`

### Test Scenarios

#### IS4.3.1: End-to-End Query Expansion
```python
import pytest
from rag_factory.strategies.query_expansion.expander_service import QueryExpanderService
from rag_factory.strategies.query_expansion.base import ExpansionConfig, ExpansionStrategy
from rag_factory.services.llm.service import LLMService, LLMConfig

@pytest.mark.integration
def test_keyword_expansion_real_llm():
    """Test keyword expansion with real LLM."""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo"
    )
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        max_additional_terms=5
    )

    service = QueryExpanderService(expansion_config, llm_service)

    result = service.expand("machine learning")

    assert result.original_query == "machine learning"
    assert len(result.primary_expansion.expanded_query) > len(result.original_query)
    assert len(result.primary_expansion.added_terms) > 0

    print(f"\nOriginal: {result.original_query}")
    print(f"Expanded: {result.primary_expansion.expanded_query}")
    print(f"Added terms: {result.primary_expansion.added_terms}")

@pytest.mark.integration
def test_query_reformulation():
    """Test query reformulation strategy."""
    llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.REFORMULATION
    )

    service = QueryExpanderService(expansion_config, llm_service)

    vague_query = "how does it work"
    result = service.expand(vague_query)

    # Reformulated query should be more specific
    assert result.primary_expansion.expanded_query != vague_query

    print(f"\nOriginal: {result.original_query}")
    print(f"Reformulated: {result.primary_expansion.expanded_query}")

@pytest.mark.integration
def test_hyde_expansion():
    """Test HyDE (Hypothetical Document Expansion)."""
    llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.HYDE,
        max_tokens=150
    )

    service = QueryExpanderService(expansion_config, llm_service)

    query = "What is the capital of France?"
    result = service.expand(query)

    # HyDE should generate a passage, not just keywords
    assert len(result.primary_expansion.expanded_query) > 50

    print(f"\nQuery: {result.original_query}")
    print(f"Hypothetical document: {result.primary_expansion.expanded_query}")

@pytest.mark.integration
def test_multi_query_generation():
    """Test generating multiple query variants."""
    llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.MULTI_QUERY,
        generate_multiple_variants=True,
        num_variants=3
    )

    service = QueryExpanderService(expansion_config, llm_service)

    result = service.expand("climate change effects")

    print(f"\nOriginal: {result.original_query}")
    print("Variants:")
    for i, variant in enumerate(result.expanded_queries, 1):
        print(f"  {i}. {variant.expanded_query}")

@pytest.mark.integration
def test_expansion_performance():
    """Test expansion performance meets requirements."""
    import time

    llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD
    )

    service = QueryExpanderService(expansion_config, llm_service)

    start = time.time()
    result = service.expand("artificial intelligence")
    duration = time.time() - start

    print(f"\nExpansion took {duration:.3f}s ({result.execution_time_ms:.0f}ms)")

    # Should meet <1 second requirement
    assert duration < 1.0, f"Expansion took {duration:.3f}s (expected <1s)"

@pytest.mark.integration
def test_ab_testing_functionality():
    """Test A/B testing capability."""
    llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        track_metrics=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    query = "neural networks"

    # Test with expansion enabled
    result_expanded = service.expand(query, enable_expansion=True)

    # Test with expansion disabled
    result_original = service.expand(query, enable_expansion=False)

    # Verify different results
    assert result_expanded.primary_expansion.expanded_query != result_original.primary_expansion.expanded_query
    assert len(result_expanded.primary_expansion.added_terms) > 0
    assert len(result_original.primary_expansion.added_terms) == 0

    # Check stats
    stats = service.get_stats()
    assert stats["expansion_enabled_count"] >= 1
    assert stats["expansion_disabled_count"] >= 1
```

---

## Definition of Done

- [ ] Base query expander interface defined
- [ ] LLM-based expander implemented
- [ ] Keyword expansion strategy implemented
- [ ] Query reformulation implemented
- [ ] Question generation implemented
- [ ] Multi-query generation implemented
- [ ] HyDE (Hypothetical Document Expansion) implemented
- [ ] Expansion prompts system working
- [ ] Caching implementation complete
- [ ] A/B testing framework working
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance meets <1s requirement
- [ ] Quality validation showing improvements
- [ ] Logging and metrics working
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Notes for Developers

1. **Start Simple**: Begin with keyword expansion before trying HyDE or multi-query.

2. **Prompt Engineering**: Spend time on prompts. Quality depends heavily on prompt quality.

3. **LLM Costs**: Query expansion uses LLM for every query. Monitor costs carefully.

4. **Caching**: Always enable caching. Same queries get expanded the same way.

5. **A/B Testing**: Run experiments to validate expansion actually improves results.

6. **HyDE**: HyDE works very well for semantic search but uses more tokens.

7. **Temperature**: Use low temperature (0.2-0.3) for consistent, focused expansions.

8. **Fallback**: Always fallback to original query on errors. Never fail user requests.

9. **Validation**: Validate expansions preserve original intent. Watch for drift.

10. **Domain Adaptation**: Use domain_context to guide expansions for specialized domains.
