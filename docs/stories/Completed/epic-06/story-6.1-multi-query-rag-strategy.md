# Story 6.1: Implement Multi-Query RAG Strategy

**Story ID:** 6.1
**Epic:** Epic 6 - Multi-Query & Contextual Strategies
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 3 (Embedding Service, LLM Service), Epic 4 (Basic retrieval)

---

## User Story

**As a** system
**I want** to generate multiple query variants
**So that** retrieval has broader coverage and captures different perspectives of the user's intent

---

## Detailed Requirements

### Functional Requirements

1. **Query Variant Generation**
   - Use LLM to generate 3-5 query variants from original query
   - Generate variants that capture different:
     - Phrasings (synonyms, alternative wording)
     - Perspectives (different angles on the question)
     - Specificity levels (more/less specific versions)
     - Related concepts (semantically related queries)
   - Configurable number of variants (min: 2, max: 10)
   - Include original query in variant set
   - Validate generated variants (non-empty, meaningful)

2. **Parallel Query Execution**
   - Execute all query variants concurrently using async/await
   - Use asyncio for parallel execution
   - Timeout handling for slow queries (configurable, default: 10s)
   - Error handling - continue if some queries fail
   - Track execution time per query
   - Support cancellation of remaining queries if enough results found

3. **Result Aggregation**
   - Collect results from all query variants
   - Merge results into single result set
   - Track which variant produced each result
   - Preserve relevance scores from each query
   - Handle empty results from some queries gracefully
   - Support configurable max results per query variant

4. **Deduplication**
   - Identify duplicate chunks across query results
   - Use chunk IDs for exact deduplication
   - Use embedding similarity for near-duplicate detection (optional)
   - Configurable similarity threshold for near-duplicates (0.9-0.99)
   - Keep highest-scoring instance of duplicates
   - Track how many variants retrieved each chunk (frequency)

5. **Result Ranking and Merging**
   - Rank merged results using multiple strategies:
     - **Maximum Score**: Take highest score across variants
     - **Average Score**: Average scores across variants
     - **Weighted Average**: Weight by variant importance
     - **Frequency Boost**: Boost chunks found by multiple variants
     - **Reciprocal Rank Fusion (RRF)**: Combine rankings using RRF
   - Configurable ranking strategy
   - Support hybrid ranking (combine multiple strategies)
   - Final top-k selection after ranking

6. **Query Variant Prompt Engineering**
   - Customizable prompt template for variant generation
   - Context-aware variant generation (domain-specific)
   - Support for different variant types:
     - Paraphrase: Same meaning, different words
     - Decompose: Break complex query into sub-queries
     - Expand: Add related concepts
     - Specify: Make more specific
     - Generalize: Make more general
   - Configurable variant mix (e.g., 2 paraphrases, 1 decompose, 1 expand)

### Non-Functional Requirements

1. **Performance**
   - Variant generation: <1s for 5 variants
   - Parallel query execution: <2s for 5 variants (assuming 100ms per query)
   - Total latency: <3s for complete multi-query retrieval
   - Support batching for multiple user queries
   - Efficient memory usage (streaming results)

2. **Reliability**
   - Handle LLM API failures gracefully
   - Fallback to original query if variant generation fails
   - Retry logic for transient failures
   - Partial results acceptable (some variants fail)
   - Minimum 1 successful query required

3. **Quality**
   - Generated variants should be semantically diverse
   - Avoid generating duplicate or near-duplicate variants
   - Variants should maintain original query intent
   - Improved recall vs single-query baseline
   - Maintain precision (avoid too much noise)

4. **Configurability**
   - All parameters configurable via config file
   - Runtime configuration overrides
   - Per-domain variant generation settings
   - Easy to add new ranking strategies
   - A/B testing support (compare strategies)

5. **Observability**
   - Log all generated variants
   - Track performance metrics (latency, success rate)
   - Monitor variant diversity (embedding distance)
   - Track result overlap between variants
   - Log ranking decisions and scores

---

## Acceptance Criteria

### AC1: Query Variant Generation
- [ ] LLM integration for variant generation working
- [ ] Generates configurable number of variants (3-5 default)
- [ ] Variants are semantically diverse (avg cosine distance > 0.1)
- [ ] Variants maintain original query intent
- [ ] Prompt template customizable
- [ ] Validation prevents empty/invalid variants
- [ ] Fallback to original query if generation fails

### AC2: Parallel Execution
- [ ] Async execution implemented using asyncio
- [ ] All variants execute concurrently
- [ ] Timeout handling working (configurable, default 10s)
- [ ] Error handling allows partial failures
- [ ] Execution time tracked per variant
- [ ] Performance: Total time ≈ max(individual times), not sum

### AC3: Result Deduplication
- [ ] Exact deduplication by chunk ID working
- [ ] Near-duplicate detection using embeddings (optional)
- [ ] Configurable similarity threshold
- [ ] Highest-scoring duplicate retained
- [ ] Frequency tracking (how many variants found each chunk)
- [ ] Deduplication stats logged

### AC4: Result Ranking
- [ ] At least 3 ranking strategies implemented:
  - Maximum Score
  - Reciprocal Rank Fusion (RRF)
  - Frequency Boost
- [ ] Strategy selection via configuration
- [ ] Hybrid ranking supported
- [ ] Final results properly ordered by score
- [ ] Top-k selection working

### AC5: Performance Requirements
- [ ] Variant generation <1s
- [ ] Parallel execution <2s for 5 variants
- [ ] Total latency <3s end-to-end
- [ ] Memory usage reasonable (<100MB overhead)
- [ ] No memory leaks in async execution

### AC6: Quality Metrics
- [ ] Improved recall vs single-query baseline (>10% improvement)
- [ ] Precision maintained (within 5% of baseline)
- [ ] Variant diversity measured and acceptable
- [ ] Result overlap analyzed and reasonable (20-40%)
- [ ] Benchmark tests demonstrate improvements

### AC7: Error Handling
- [ ] LLM failures handled gracefully
- [ ] Fallback to original query working
- [ ] Partial results acceptable
- [ ] Retry logic implemented
- [ ] Error logging comprehensive

### AC8: Testing
- [ ] Unit tests for all components (>90% coverage)
- [ ] Integration tests with real LLM and vector store
- [ ] Performance benchmarks meet requirements
- [ ] Quality comparison tests vs baseline
- [ ] Edge case testing (timeouts, failures, empty results)

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── multi_query/
│   │   ├── __init__.py
│   │   ├── strategy.py              # Main multi-query strategy
│   │   ├── variant_generator.py     # LLM-based query variant generation
│   │   ├── parallel_executor.py     # Async parallel query execution
│   │   ├── deduplicator.py          # Result deduplication
│   │   ├── ranker.py                # Result ranking strategies
│   │   ├── config.py                # Configuration models
│   │   └── prompts.py               # Prompt templates

tests/
├── unit/
│   └── strategies/
│       └── multi_query/
│           ├── test_variant_generator.py
│           ├── test_parallel_executor.py
│           ├── test_deduplicator.py
│           └── test_ranker.py
│
├── integration/
│   └── strategies/
│       └── test_multi_query_integration.py
│
├── benchmarks/
│   └── test_multi_query_performance.py
```

### Dependencies
```python
# requirements.txt additions
asyncio>=3.4.3            # Async parallel execution (built-in)
aiohttp>=3.8.0            # Async HTTP for API calls
tenacity>=8.0.0           # Retry logic
```

### Configuration Models
```python
# rag_factory/strategies/multi_query/config.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class VariantType(Enum):
    """Types of query variants to generate."""
    PARAPHRASE = "paraphrase"
    DECOMPOSE = "decompose"
    EXPAND = "expand"
    SPECIFY = "specify"
    GENERALIZE = "generalize"

class RankingStrategy(Enum):
    """Strategies for ranking merged results."""
    MAX_SCORE = "max_score"
    AVERAGE_SCORE = "average_score"
    WEIGHTED_AVERAGE = "weighted_average"
    FREQUENCY_BOOST = "frequency_boost"
    RECIPROCAL_RANK_FUSION = "rrf"
    HYBRID = "hybrid"

class MultiQueryConfig(BaseModel):
    """Configuration for multi-query RAG strategy."""

    # Variant generation
    num_variants: int = Field(default=3, ge=2, le=10, description="Number of query variants to generate")
    variant_types: List[VariantType] = Field(
        default=[VariantType.PARAPHRASE, VariantType.EXPAND],
        description="Types of variants to generate"
    )
    include_original: bool = Field(default=True, description="Include original query in variants")

    # LLM settings
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model for variant generation")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for variant generation")
    variant_generation_timeout: float = Field(default=5.0, description="Timeout for variant generation (seconds)")

    # Parallel execution
    query_timeout: float = Field(default=10.0, description="Timeout per query variant (seconds)")
    max_concurrent_queries: int = Field(default=10, description="Max concurrent query executions")

    # Results per variant
    top_k_per_variant: int = Field(default=10, description="Results to retrieve per variant")

    # Deduplication
    enable_near_duplicate_detection: bool = Field(default=False, description="Enable embedding-based near-duplicate detection")
    near_duplicate_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Similarity threshold for near-duplicates")

    # Ranking
    ranking_strategy: RankingStrategy = Field(default=RankingStrategy.RECIPROCAL_RANK_FUSION, description="Result ranking strategy")
    frequency_boost_weight: float = Field(default=0.2, description="Weight for frequency boost in ranking")
    rrf_k: int = Field(default=60, description="K parameter for Reciprocal Rank Fusion")

    # Output
    final_top_k: int = Field(default=5, description="Final number of results to return")

    # Fallback
    fallback_to_original: bool = Field(default=True, description="Fallback to original query if variant generation fails")
    min_successful_queries: int = Field(default=1, description="Minimum successful queries required")

    # Observability
    log_variants: bool = Field(default=True, description="Log generated variants")
    track_metrics: bool = Field(default=True, description="Track performance metrics")
```

### Variant Generator
```python
# rag_factory/strategies/multi_query/variant_generator.py
from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import MultiQueryConfig, VariantType
from .prompts import VARIANT_GENERATION_PROMPTS
from ...services.llm import LLMService

logger = logging.getLogger(__name__)

class QueryVariantGenerator:
    """Generates query variants using LLM."""

    def __init__(self, llm_service: LLMService, config: MultiQueryConfig):
        """
        Initialize variant generator.

        Args:
            llm_service: LLM service for generation
            config: Multi-query configuration
        """
        self.llm_service = llm_service
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_variants(self, query: str) -> List[str]:
        """
        Generate query variants from original query.

        Args:
            query: Original user query

        Returns:
            List of query variants (including original if configured)
        """
        logger.info(f"Generating {self.config.num_variants} variants for query: {query}")

        try:
            # Build prompt for variant generation
            prompt = self._build_variant_prompt(query)

            # Generate variants using LLM
            response = await self.llm_service.agenerate(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=500,
                timeout=self.config.variant_generation_timeout
            )

            # Parse variants from response
            variants = self._parse_variants(response.text, query)

            # Validate variants
            variants = self._validate_variants(variants, query)

            # Include original query if configured
            if self.config.include_original and query not in variants:
                variants.insert(0, query)

            logger.info(f"Generated {len(variants)} variants: {variants}")

            return variants

        except Exception as e:
            logger.error(f"Error generating variants: {e}")

            # Fallback to original query
            if self.config.fallback_to_original:
                logger.info("Falling back to original query")
                return [query]
            else:
                raise

    def _build_variant_prompt(self, query: str) -> str:
        """Build prompt for variant generation based on configured types."""
        variant_types_str = ", ".join([vt.value for vt in self.config.variant_types])

        prompt = f"""Generate {self.config.num_variants} diverse variants of the following query.
Each variant should capture the same intent but use different phrasings, perspectives, or levels of specificity.

Generate these types of variants: {variant_types_str}

Original Query: {query}

Generate exactly {self.config.num_variants} variants, one per line:"""

        return prompt

    def _parse_variants(self, response: str, original_query: str) -> List[str]:
        """Parse variants from LLM response."""
        lines = response.strip().split("\n")

        variants = []
        for line in lines:
            # Clean up line (remove numbering, bullets, etc.)
            cleaned = line.strip()
            cleaned = cleaned.lstrip("0123456789.-) ")

            if cleaned and len(cleaned) > 5:  # Skip very short lines
                variants.append(cleaned)

        # Limit to requested number
        return variants[:self.config.num_variants]

    def _validate_variants(self, variants: List[str], original_query: str) -> List[str]:
        """Validate generated variants."""
        validated = []

        for variant in variants:
            # Check not empty
            if not variant or len(variant.strip()) < 5:
                continue

            # Check not duplicate
            if variant in validated:
                continue

            # Check not too similar to already validated variants
            # (Simple check - could use embeddings for better validation)
            if any(variant.lower() == v.lower() for v in validated):
                continue

            validated.append(variant)

        return validated
```

### Parallel Executor
```python
# rag_factory/strategies/multi_query/parallel_executor.py
import asyncio
from typing import List, Dict, Any
import logging
import time
from .config import MultiQueryConfig

logger = logging.getLogger(__name__)

class ParallelQueryExecutor:
    """Executes multiple query variants in parallel."""

    def __init__(self, vector_store_service: Any, config: MultiQueryConfig):
        """
        Initialize parallel executor.

        Args:
            vector_store_service: Vector store for querying
            config: Multi-query configuration
        """
        self.vector_store = vector_store_service
        self.config = config

    async def execute_queries(self, query_variants: List[str]) -> List[Dict[str, Any]]:
        """
        Execute all query variants in parallel.

        Args:
            query_variants: List of query variants to execute

        Returns:
            List of dicts with query results and metadata
        """
        logger.info(f"Executing {len(query_variants)} queries in parallel")

        # Create tasks for all queries
        tasks = []
        for i, query in enumerate(query_variants):
            task = self._execute_single_query(query, variant_index=i)
            tasks.append(task)

        # Execute all queries concurrently with timeout
        start_time = time.time()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time
        logger.info(f"Parallel execution completed in {execution_time:.2f}s")

        # Process results (handle exceptions)
        processed_results = []
        successful_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Query variant {i} failed: {result}")
                # Add empty result
                processed_results.append({
                    "query": query_variants[i],
                    "variant_index": i,
                    "results": [],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                successful_count += 1

        # Check minimum successful queries
        if successful_count < self.config.min_successful_queries:
            raise ValueError(
                f"Only {successful_count} queries succeeded, "
                f"minimum required: {self.config.min_successful_queries}"
            )

        logger.info(f"{successful_count}/{len(query_variants)} queries succeeded")

        return processed_results

    async def _execute_single_query(
        self,
        query: str,
        variant_index: int
    ) -> Dict[str, Any]:
        """
        Execute a single query with timeout.

        Args:
            query: Query string
            variant_index: Index of this variant

        Returns:
            Dict with query results and metadata
        """
        logger.debug(f"Executing variant {variant_index}: {query}")

        start_time = time.time()

        try:
            # Execute query with timeout
            results = await asyncio.wait_for(
                self._query_vector_store(query),
                timeout=self.config.query_timeout
            )

            execution_time = time.time() - start_time

            return {
                "query": query,
                "variant_index": variant_index,
                "results": results,
                "success": True,
                "execution_time": execution_time,
                "num_results": len(results)
            }

        except asyncio.TimeoutError:
            logger.warning(f"Query variant {variant_index} timed out after {self.config.query_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Error executing variant {variant_index}: {e}")
            raise

    async def _query_vector_store(self, query: str) -> List[Dict[str, Any]]:
        """Query vector store (async wrapper)."""
        # If vector store has async support, use it directly
        if hasattr(self.vector_store, 'asearch'):
            return await self.vector_store.asearch(
                query=query,
                top_k=self.config.top_k_per_variant
            )
        else:
            # Otherwise, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.vector_store.search,
                query,
                self.config.top_k_per_variant
            )
```

### Deduplicator
```python
# rag_factory/strategies/multi_query/deduplicator.py
from typing import List, Dict, Any, Set
import logging
import numpy as np
from .config import MultiQueryConfig

logger = logging.getLogger(__name__)

class ResultDeduplicator:
    """Deduplicates results from multiple query variants."""

    def __init__(self, config: MultiQueryConfig, embedding_service: Any = None):
        """
        Initialize deduplicator.

        Args:
            config: Multi-query configuration
            embedding_service: Optional embedding service for near-duplicate detection
        """
        self.config = config
        self.embedding_service = embedding_service

    def deduplicate(
        self,
        query_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate results across query variants.

        Args:
            query_results: List of result dicts from parallel execution

        Returns:
            Deduplicated list of results with frequency tracking
        """
        logger.info("Deduplicating results from multiple variants")

        # Track unique chunks
        chunk_map: Dict[str, Dict[str, Any]] = {}
        chunk_frequency: Dict[str, int] = {}
        chunk_max_score: Dict[str, float] = {}
        chunk_variant_indices: Dict[str, List[int]] = {}

        # Collect all results
        for query_result in query_results:
            if not query_result.get("success", False):
                continue

            variant_index = query_result["variant_index"]
            results = query_result.get("results", [])

            for result in results:
                chunk_id = result.get("chunk_id") or result.get("id")

                if not chunk_id:
                    continue

                score = result.get("score", 0.0)

                # Track first occurrence
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = result
                    chunk_frequency[chunk_id] = 1
                    chunk_max_score[chunk_id] = score
                    chunk_variant_indices[chunk_id] = [variant_index]
                else:
                    # Update frequency and max score
                    chunk_frequency[chunk_id] += 1
                    chunk_max_score[chunk_id] = max(chunk_max_score[chunk_id], score)
                    chunk_variant_indices[chunk_id].append(variant_index)

        # Near-duplicate detection (optional)
        if self.config.enable_near_duplicate_detection and self.embedding_service:
            chunk_map = self._remove_near_duplicates(chunk_map, chunk_max_score)

        # Build deduplicated results with metadata
        deduplicated = []
        for chunk_id, chunk in chunk_map.items():
            deduplicated.append({
                **chunk,
                "frequency": chunk_frequency[chunk_id],
                "max_score": chunk_max_score[chunk_id],
                "variant_indices": chunk_variant_indices[chunk_id],
                "found_by_variants": len(chunk_variant_indices[chunk_id])
            })

        logger.info(
            f"Deduplicated {sum(len(qr.get('results', [])) for qr in query_results)} "
            f"results to {len(deduplicated)} unique chunks"
        )

        return deduplicated

    def _remove_near_duplicates(
        self,
        chunk_map: Dict[str, Dict[str, Any]],
        chunk_max_score: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Remove near-duplicate chunks based on embedding similarity."""
        logger.info("Detecting near-duplicates using embeddings")

        chunk_ids = list(chunk_map.keys())

        if len(chunk_ids) < 2:
            return chunk_map

        # Get embeddings for all chunks
        texts = [chunk_map[cid].get("text", "") for cid in chunk_ids]

        try:
            embedding_result = self.embedding_service.embed(texts)
            embeddings = embedding_result.embeddings

            # Find near-duplicates
            to_remove: Set[str] = set()

            for i in range(len(chunk_ids)):
                if chunk_ids[i] in to_remove:
                    continue

                for j in range(i + 1, len(chunk_ids)):
                    if chunk_ids[j] in to_remove:
                        continue

                    # Calculate similarity
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                    if similarity >= self.config.near_duplicate_threshold:
                        # Keep the one with higher score
                        if chunk_max_score[chunk_ids[i]] >= chunk_max_score[chunk_ids[j]]:
                            to_remove.add(chunk_ids[j])
                        else:
                            to_remove.add(chunk_ids[i])
                            break  # Move to next i

            # Remove near-duplicates
            for chunk_id in to_remove:
                del chunk_map[chunk_id]

            logger.info(f"Removed {len(to_remove)} near-duplicate chunks")

        except Exception as e:
            logger.warning(f"Near-duplicate detection failed: {e}, skipping")

        return chunk_map

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### Result Ranker
```python
# rag_factory/strategies/multi_query/ranker.py
from typing import List, Dict, Any
import logging
import numpy as np
from .config import MultiQueryConfig, RankingStrategy

logger = logging.getLogger(__name__)

class ResultRanker:
    """Ranks merged results from multiple query variants."""

    def __init__(self, config: MultiQueryConfig):
        """
        Initialize result ranker.

        Args:
            config: Multi-query configuration
        """
        self.config = config

    def rank(self, deduplicated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank deduplicated results using configured strategy.

        Args:
            deduplicated_results: Deduplicated results with frequency metadata

        Returns:
            Ranked and scored results
        """
        logger.info(f"Ranking {len(deduplicated_results)} results using {self.config.ranking_strategy.value}")

        if not deduplicated_results:
            return []

        # Apply ranking strategy
        if self.config.ranking_strategy == RankingStrategy.MAX_SCORE:
            ranked = self._rank_by_max_score(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.AVERAGE_SCORE:
            ranked = self._rank_by_average_score(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.FREQUENCY_BOOST:
            ranked = self._rank_by_frequency_boost(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.RECIPROCAL_RANK_FUSION:
            ranked = self._rank_by_rrf(deduplicated_results)
        elif self.config.ranking_strategy == RankingStrategy.HYBRID:
            ranked = self._rank_hybrid(deduplicated_results)
        else:
            # Default to max score
            ranked = self._rank_by_max_score(deduplicated_results)

        # Sort by final score
        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        # Select top-k
        top_k = ranked[:self.config.final_top_k]

        logger.info(f"Returning top {len(top_k)} results")

        return top_k

    def _rank_by_max_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by maximum score across variants."""
        for result in results:
            result["final_score"] = result.get("max_score", 0.0)
            result["ranking_method"] = "max_score"
        return results

    def _rank_by_average_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by average score (placeholder - would need all scores)."""
        # This is simplified - in practice, would track all scores per variant
        for result in results:
            result["final_score"] = result.get("max_score", 0.0)
            result["ranking_method"] = "average_score"
        return results

    def _rank_by_frequency_boost(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank with frequency boost (chunks found by multiple variants ranked higher)."""
        for result in results:
            base_score = result.get("max_score", 0.0)
            frequency = result.get("frequency", 1)

            # Boost score based on frequency
            frequency_boost = 1.0 + (frequency - 1) * self.config.frequency_boost_weight
            result["final_score"] = base_score * frequency_boost
            result["ranking_method"] = "frequency_boost"
            result["frequency_boost_factor"] = frequency_boost

        return results

    def _rank_by_rrf(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for all variants
        where rank_i is the rank of this document in variant i
        """
        # Group results by variant
        variant_rankings: Dict[int, List[Dict[str, Any]]] = {}

        for result in results:
            for variant_idx in result.get("variant_indices", []):
                if variant_idx not in variant_rankings:
                    variant_rankings[variant_idx] = []
                variant_rankings[variant_idx].append(result)

        # Sort each variant's results by score
        for variant_idx in variant_rankings:
            variant_rankings[variant_idx].sort(
                key=lambda x: x.get("max_score", 0.0),
                reverse=True
            )

        # Calculate RRF scores
        chunk_rrf_scores: Dict[str, float] = {}

        for variant_idx, variant_results in variant_rankings.items():
            for rank, result in enumerate(variant_results, start=1):
                chunk_id = result.get("chunk_id") or result.get("id")
                rrf_contribution = 1.0 / (self.config.rrf_k + rank)

                if chunk_id not in chunk_rrf_scores:
                    chunk_rrf_scores[chunk_id] = 0.0
                chunk_rrf_scores[chunk_id] += rrf_contribution

        # Assign RRF scores to results
        for result in results:
            chunk_id = result.get("chunk_id") or result.get("id")
            result["final_score"] = chunk_rrf_scores.get(chunk_id, 0.0)
            result["ranking_method"] = "reciprocal_rank_fusion"

        return results

    def _rank_hybrid(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hybrid ranking combining RRF and frequency boost."""
        # Apply RRF
        results = self._rank_by_rrf(results)

        # Apply frequency boost on top of RRF
        for result in results:
            rrf_score = result["final_score"]
            frequency = result.get("frequency", 1)
            frequency_boost = 1.0 + (frequency - 1) * self.config.frequency_boost_weight

            result["final_score"] = rrf_score * frequency_boost
            result["ranking_method"] = "hybrid_rrf_frequency"

        return results
```

### Main Strategy Implementation
```python
# rag_factory/strategies/multi_query/strategy.py
from typing import List, Dict, Any, Optional
import logging
import asyncio
from ..base import RAGStrategy
from .config import MultiQueryConfig
from .variant_generator import QueryVariantGenerator
from .parallel_executor import ParallelQueryExecutor
from .deduplicator import ResultDeduplicator
from .ranker import ResultRanker

logger = logging.getLogger(__name__)

class MultiQueryRAGStrategy(RAGStrategy):
    """
    Multi-Query RAG: Generate multiple query variants and merge results.

    This strategy:
    1. Generates 3-5 query variants using LLM
    2. Executes all variants in parallel
    3. Deduplicates results across variants
    4. Ranks merged results using RRF or other strategies
    """

    def __init__(
        self,
        vector_store_service: Any,
        llm_service: Any,
        embedding_service: Any = None,
        config: Optional[MultiQueryConfig] = None
    ):
        """
        Initialize multi-query strategy.

        Args:
            vector_store_service: Vector store for retrieval
            llm_service: LLM service for variant generation
            embedding_service: Optional embedding service for near-duplicate detection
            config: Multi-query configuration
        """
        self.vector_store = vector_store_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.config = config or MultiQueryConfig()

        # Initialize components
        self.variant_generator = QueryVariantGenerator(llm_service, self.config)
        self.parallel_executor = ParallelQueryExecutor(vector_store_service, self.config)
        self.deduplicator = ResultDeduplicator(self.config, embedding_service)
        self.ranker = ResultRanker(self.config)

    async def aretrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Async retrieve with multi-query strategy.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            List of ranked results
        """
        logger.info(f"Multi-query retrieval for: {query}")

        try:
            # Step 1: Generate query variants
            variants = await self.variant_generator.generate_variants(query)

            if self.config.log_variants:
                logger.info(f"Query variants: {variants}")

            # Step 2: Execute variants in parallel
            query_results = await self.parallel_executor.execute_queries(variants)

            # Step 3: Deduplicate results
            deduplicated = self.deduplicator.deduplicate(query_results)

            # Step 4: Rank merged results
            ranked = self.ranker.rank(deduplicated)

            logger.info(f"Multi-query retrieval complete: {len(ranked)} results")

            return ranked

        except Exception as e:
            logger.error(f"Multi-query retrieval failed: {e}")

            # Fallback to single query
            if self.config.fallback_to_original:
                logger.info("Falling back to single-query retrieval")
                return await self._fallback_retrieve(query)
            else:
                raise

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Synchronous retrieve (wrapper for async).

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            List of ranked results
        """
        # Run async retrieve in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new task
            return asyncio.create_task(self.aretrieve(query, **kwargs))
        else:
            # Run in new event loop
            return loop.run_until_complete(self.aretrieve(query, **kwargs))

    async def _fallback_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to single query retrieval."""
        if hasattr(self.vector_store, 'asearch'):
            return await self.vector_store.asearch(
                query=query,
                top_k=self.config.final_top_k
            )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.vector_store.search,
                query,
                self.config.final_top_k
            )

    @property
    def name(self) -> str:
        return "multi_query"

    @property
    def description(self) -> str:
        return "Generate multiple query variants and merge results for broader coverage"
```

---

## Unit Tests

### Test File Locations
- `tests/unit/strategies/multi_query/test_variant_generator.py`
- `tests/unit/strategies/multi_query/test_parallel_executor.py`
- `tests/unit/strategies/multi_query/test_deduplicator.py`
- `tests/unit/strategies/multi_query/test_ranker.py`

### Test Cases

#### TC6.1.1: Variant Generator Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from rag_factory.strategies.multi_query.variant_generator import QueryVariantGenerator
from rag_factory.strategies.multi_query.config import MultiQueryConfig

@pytest.fixture
def mock_llm_service():
    service = Mock()
    # Mock async generate
    response = Mock()
    response.text = """What is machine learning?
How does machine learning work?
Explain machine learning concepts
Machine learning definition"""
    service.agenerate = AsyncMock(return_value=response)
    return service

@pytest.fixture
def config():
    return MultiQueryConfig(num_variants=3)

@pytest.fixture
def variant_generator(mock_llm_service, config):
    return QueryVariantGenerator(mock_llm_service, config)

@pytest.mark.asyncio
async def test_generate_variants_basic(variant_generator, mock_llm_service):
    """Test basic variant generation."""
    query = "What is machine learning?"

    variants = await variant_generator.generate_variants(query)

    # Should generate requested number of variants
    assert len(variants) >= 3
    assert query in variants  # Original included
    mock_llm_service.agenerate.assert_called_once()

@pytest.mark.asyncio
async def test_generate_variants_include_original(mock_llm_service):
    """Test that original query is included when configured."""
    config = MultiQueryConfig(num_variants=3, include_original=True)
    generator = QueryVariantGenerator(mock_llm_service, config)

    query = "What is AI?"
    variants = await generator.generate_variants(query)

    assert query in variants

@pytest.mark.asyncio
async def test_generate_variants_exclude_original(mock_llm_service):
    """Test that original query can be excluded."""
    config = MultiQueryConfig(num_variants=3, include_original=False)
    generator = QueryVariantGenerator(mock_llm_service, config)

    query = "What is AI?"

    # Mock response without original query
    response = Mock()
    response.text = "How does AI work?\nExplain artificial intelligence\nAI definition"
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    variants = await generator.generate_variants(query)

    # Original might still be in parsed variants, so check length
    assert len(variants) <= 4

@pytest.mark.asyncio
async def test_variant_validation(variant_generator, mock_llm_service):
    """Test variant validation removes invalid variants."""
    # Mock response with some invalid variants
    response = Mock()
    response.text = """Valid variant 1

    Valid variant 2
    short
    Valid variant 3"""
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    query = "Test query"
    variants = await variant_generator.generate_variants(query)

    # Should filter out empty lines and very short variants
    assert all(len(v) > 5 for v in variants if v != query)

@pytest.mark.asyncio
async def test_variant_generation_failure_fallback(mock_llm_service):
    """Test fallback to original query on generation failure."""
    config = MultiQueryConfig(fallback_to_original=True)
    generator = QueryVariantGenerator(mock_llm_service, config)

    # Mock LLM failure
    mock_llm_service.agenerate = AsyncMock(side_effect=Exception("LLM error"))

    query = "Test query"
    variants = await generator.generate_variants(query)

    # Should fall back to original query
    assert variants == [query]

@pytest.mark.asyncio
async def test_variant_deduplication(variant_generator, mock_llm_service):
    """Test that duplicate variants are removed."""
    # Mock response with duplicates
    response = Mock()
    response.text = """Variant 1
Variant 2
Variant 1
Variant 3"""
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    query = "Test"
    variants = await variant_generator.generate_variants(query)

    # Should remove duplicates
    assert len(variants) == len(set(variants))
```

#### TC6.1.2: Parallel Executor Tests
```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.multi_query.parallel_executor import ParallelQueryExecutor
from rag_factory.strategies.multi_query.config import MultiQueryConfig

@pytest.fixture
def mock_vector_store():
    store = Mock()

    async def mock_search(query, top_k):
        # Simulate different results for different queries
        await asyncio.sleep(0.1)  # Simulate query time
        return [
            {"chunk_id": f"chunk_{i}", "text": f"Result {i} for {query}", "score": 0.9 - i * 0.1}
            for i in range(min(3, top_k))
        ]

    store.asearch = mock_search
    return store

@pytest.fixture
def config():
    return MultiQueryConfig(
        top_k_per_variant=5,
        query_timeout=5.0
    )

@pytest.fixture
def executor(mock_vector_store, config):
    return ParallelQueryExecutor(mock_vector_store, config)

@pytest.mark.asyncio
async def test_execute_queries_parallel(executor):
    """Test parallel execution of multiple queries."""
    variants = ["query 1", "query 2", "query 3"]

    import time
    start = time.time()
    results = await executor.execute_queries(variants)
    duration = time.time() - start

    # Should execute in parallel (close to max time, not sum)
    assert duration < 0.5  # Much less than 0.3s (3 * 0.1s)

    # Should get results for all variants
    assert len(results) == 3
    assert all(r["success"] for r in results)

@pytest.mark.asyncio
async def test_execute_single_query(executor):
    """Test single query execution."""
    result = await executor._execute_single_query("test query", variant_index=0)

    assert result["success"] is True
    assert result["variant_index"] == 0
    assert result["query"] == "test query"
    assert "results" in result
    assert "execution_time" in result

@pytest.mark.asyncio
async def test_query_timeout_handling(mock_vector_store):
    """Test timeout handling for slow queries."""
    # Mock slow query
    async def slow_search(query, top_k):
        await asyncio.sleep(10)  # Simulate very slow query
        return []

    mock_vector_store.asearch = slow_search

    config = MultiQueryConfig(query_timeout=0.5)
    executor = ParallelQueryExecutor(mock_vector_store, config)

    variants = ["slow query"]
    results = await executor.execute_queries(variants)

    # Should handle timeout gracefully
    assert len(results) == 1
    assert results[0]["success"] is False
    assert "error" in results[0]

@pytest.mark.asyncio
async def test_partial_failure_handling(mock_vector_store):
    """Test handling of partial failures."""
    call_count = 0

    async def flaky_search(query, top_k):
        nonlocal call_count
        call_count += 1

        if call_count == 2:
            raise Exception("Query failed")

        return [{"chunk_id": "chunk_1", "text": "Result", "score": 0.9}]

    mock_vector_store.asearch = flaky_search

    executor = ParallelQueryExecutor(mock_vector_store, MultiQueryConfig())

    variants = ["query 1", "query 2", "query 3"]
    results = await executor.execute_queries(variants)

    # Should handle one failure
    assert len(results) == 3
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    assert len(successful) == 2
    assert len(failed) == 1

@pytest.mark.asyncio
async def test_minimum_successful_queries(mock_vector_store):
    """Test minimum successful queries requirement."""
    # Mock all queries failing
    async def failing_search(query, top_k):
        raise Exception("All queries fail")

    mock_vector_store.asearch = failing_search

    config = MultiQueryConfig(min_successful_queries=2)
    executor = ParallelQueryExecutor(mock_vector_store, config)

    variants = ["query 1", "query 2"]

    # Should raise error when minimum not met
    with pytest.raises(ValueError, match="minimum required"):
        await executor.execute_queries(variants)
```

#### TC6.1.3: Deduplicator Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.strategies.multi_query.deduplicator import ResultDeduplicator
from rag_factory.strategies.multi_query.config import MultiQueryConfig

@pytest.fixture
def config():
    return MultiQueryConfig()

@pytest.fixture
def deduplicator(config):
    return ResultDeduplicator(config)

def test_exact_deduplication(deduplicator):
    """Test exact deduplication by chunk ID."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [
                {"chunk_id": "chunk_1", "text": "Text 1", "score": 0.9},
                {"chunk_id": "chunk_2", "text": "Text 2", "score": 0.8}
            ]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [
                {"chunk_id": "chunk_1", "text": "Text 1", "score": 0.85},  # Duplicate
                {"chunk_id": "chunk_3", "text": "Text 3", "score": 0.7}
            ]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    # Should have 3 unique chunks
    assert len(deduplicated) == 3

    # Check chunk_1 was deduplicated
    chunk_1 = next(c for c in deduplicated if c["chunk_id"] == "chunk_1")
    assert chunk_1["frequency"] == 2  # Found by 2 variants
    assert chunk_1["max_score"] == 0.9  # Higher score kept
    assert len(chunk_1["variant_indices"]) == 2

def test_frequency_tracking(deduplicator):
    """Test frequency tracking for chunks found by multiple variants."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "text": "Text", "score": 0.9}]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "text": "Text", "score": 0.85}]
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "text": "Text", "score": 0.8}]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    assert len(deduplicated) == 1
    assert deduplicated[0]["frequency"] == 3
    assert deduplicated[0]["found_by_variants"] == 3
    assert deduplicated[0]["max_score"] == 0.9

def test_max_score_retention(deduplicator):
    """Test that highest score is retained for duplicates."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "score": 0.7}]
        },
        {
            "variant_index": 1,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "score": 0.95}]
        },
        {
            "variant_index": 2,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "score": 0.8}]
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    assert deduplicated[0]["max_score"] == 0.95

def test_skip_failed_queries(deduplicator):
    """Test that failed queries are skipped."""
    query_results = [
        {
            "variant_index": 0,
            "success": True,
            "results": [{"chunk_id": "chunk_1", "score": 0.9}]
        },
        {
            "variant_index": 1,
            "success": False,  # Failed query
            "results": []
        }
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    assert len(deduplicated) == 1
    assert deduplicated[0]["found_by_variants"] == 1

def test_empty_results(deduplicator):
    """Test handling of empty results."""
    query_results = [
        {"variant_index": 0, "success": True, "results": []},
        {"variant_index": 1, "success": True, "results": []}
    ]

    deduplicated = deduplicator.deduplicate(query_results)

    assert deduplicated == []
```

#### TC6.1.4: Ranker Tests
```python
import pytest
from rag_factory.strategies.multi_query.ranker import ResultRanker
from rag_factory.strategies.multi_query.config import MultiQueryConfig, RankingStrategy

@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "chunk_1",
            "text": "Result 1",
            "max_score": 0.9,
            "frequency": 3,
            "variant_indices": [0, 1, 2]
        },
        {
            "chunk_id": "chunk_2",
            "text": "Result 2",
            "max_score": 0.95,
            "frequency": 1,
            "variant_indices": [0]
        },
        {
            "chunk_id": "chunk_3",
            "text": "Result 3",
            "max_score": 0.85,
            "frequency": 2,
            "variant_indices": [1, 2]
        }
    ]

def test_rank_by_max_score(sample_results):
    """Test ranking by maximum score."""
    config = MultiQueryConfig(ranking_strategy=RankingStrategy.MAX_SCORE, final_top_k=10)
    ranker = ResultRanker(config)

    ranked = ranker.rank(sample_results)

    # Should be sorted by max_score (descending)
    assert ranked[0]["chunk_id"] == "chunk_2"  # 0.95
    assert ranked[1]["chunk_id"] == "chunk_1"  # 0.9
    assert ranked[2]["chunk_id"] == "chunk_3"  # 0.85

    assert all(r["ranking_method"] == "max_score" for r in ranked)

def test_rank_by_frequency_boost(sample_results):
    """Test ranking with frequency boost."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.FREQUENCY_BOOST,
        frequency_boost_weight=0.2,
        final_top_k=10
    )
    ranker = ResultRanker(config)

    ranked = ranker.rank(sample_results)

    # chunk_1 has lower base score (0.9) but frequency 3
    # With boost: 0.9 * (1 + 2 * 0.2) = 0.9 * 1.4 = 1.26
    # chunk_2: 0.95 * 1 = 0.95
    # chunk_1 should rank higher due to frequency boost

    assert ranked[0]["chunk_id"] == "chunk_1"
    assert "frequency_boost_factor" in ranked[0]

def test_rank_by_rrf(sample_results):
    """Test reciprocal rank fusion ranking."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
        rrf_k=60,
        final_top_k=10
    )
    ranker = ResultRanker(config)

    ranked = ranker.rank(sample_results)

    # RRF should give high scores to chunks found by multiple variants
    assert all(r["ranking_method"] == "reciprocal_rank_fusion" for r in ranked)
    assert all("final_score" in r for r in ranked)

def test_top_k_selection(sample_results):
    """Test top-k selection."""
    config = MultiQueryConfig(ranking_strategy=RankingStrategy.MAX_SCORE, final_top_k=2)
    ranker = ResultRanker(config)

    ranked = ranker.rank(sample_results)

    # Should return only top 2
    assert len(ranked) == 2
    assert ranked[0]["max_score"] >= ranked[1]["max_score"]

def test_empty_results():
    """Test handling of empty results."""
    config = MultiQueryConfig()
    ranker = ResultRanker(config)

    ranked = ranker.rank([])

    assert ranked == []

def test_hybrid_ranking(sample_results):
    """Test hybrid ranking strategy."""
    config = MultiQueryConfig(
        ranking_strategy=RankingStrategy.HYBRID,
        frequency_boost_weight=0.2,
        final_top_k=10
    )
    ranker = ResultRanker(config)

    ranked = ranker.rank(sample_results)

    assert all(r["ranking_method"] == "hybrid_rrf_frequency" for r in ranked)
    # Hybrid should combine RRF and frequency boost
    assert ranked[0]["chunk_id"] == "chunk_1"  # High frequency should win
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_multi_query_integration.py`

### Test Scenarios

#### IS6.1.1: End-to-End Multi-Query Workflow
```python
import pytest
import asyncio
from rag_factory.strategies.multi_query.strategy import MultiQueryRAGStrategy
from rag_factory.strategies.multi_query.config import MultiQueryConfig, RankingStrategy

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_query_complete_workflow(test_vector_store, test_llm_service, test_embedding_service):
    """Test complete multi-query RAG workflow."""
    # Setup strategy
    config = MultiQueryConfig(
        num_variants=3,
        ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
        final_top_k=5
    )

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        embedding_service=test_embedding_service,
        config=config
    )

    # Execute multi-query retrieval
    query = "What is machine learning?"
    results = await strategy.aretrieve(query)

    # Assertions
    assert len(results) <= 5
    assert all("final_score" in r for r in results)
    assert all("frequency" in r for r in results)
    assert all("ranking_method" in r for r in results)

    # Results should be sorted by score
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.integration
def test_multi_query_sync_wrapper(test_vector_store, test_llm_service):
    """Test synchronous wrapper for multi-query retrieval."""
    config = MultiQueryConfig(num_variants=2, final_top_k=3)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        config=config
    )

    query = "Test query"
    results = strategy.retrieve(query)

    assert len(results) <= 3
    assert all(isinstance(r, dict) for r in results)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_variant_diversity(test_vector_store, test_llm_service, test_embedding_service):
    """Test that generated variants are semantically diverse."""
    config = MultiQueryConfig(num_variants=5, log_variants=True)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        embedding_service=test_embedding_service,
        config=config
    )

    query = "Explain neural networks"

    # Generate variants
    variants = await strategy.variant_generator.generate_variants(query)

    # Calculate diversity using embeddings
    if len(variants) > 1:
        embed_result = test_embedding_service.embed(variants)
        embeddings = embed_result.embeddings

        # Calculate pairwise similarities
        import numpy as np
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Variants should not be too similar (avg < 0.9 indicates diversity)
        assert avg_similarity < 0.9, f"Variants too similar: {avg_similarity}"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_performance_requirements(test_vector_store, test_llm_service):
    """Test that performance requirements are met."""
    import time

    config = MultiQueryConfig(num_variants=5, final_top_k=5)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        config=config
    )

    query = "What is deep learning?"

    start = time.time()
    results = await strategy.aretrieve(query)
    duration = time.time() - start

    # Should complete in < 3 seconds
    assert duration < 3.0, f"Took {duration:.2f}s, expected <3s"

    print(f"\nMulti-query retrieval completed in {duration:.2f}s")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fallback_on_failure(test_vector_store):
    """Test fallback to original query when variant generation fails."""
    # Mock LLM service that fails
    mock_llm = Mock()
    mock_llm.agenerate = AsyncMock(side_effect=Exception("LLM failure"))

    config = MultiQueryConfig(fallback_to_original=True)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=mock_llm,
        config=config
    )

    query = "Test query"
    results = await strategy.aretrieve(query)

    # Should still return results via fallback
    assert len(results) > 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_ranking_strategy_comparison(test_vector_store, test_llm_service):
    """Compare different ranking strategies."""
    query = "What is artificial intelligence?"

    strategies_to_test = [
        RankingStrategy.MAX_SCORE,
        RankingStrategy.FREQUENCY_BOOST,
        RankingStrategy.RECIPROCAL_RANK_FUSION
    ]

    results_by_strategy = {}

    for ranking_strat in strategies_to_test:
        config = MultiQueryConfig(
            num_variants=3,
            ranking_strategy=ranking_strat,
            final_top_k=5
        )

        strategy = MultiQueryRAGStrategy(
            vector_store_service=test_vector_store,
            llm_service=test_llm_service,
            config=config
        )

        results = await strategy.aretrieve(query)
        results_by_strategy[ranking_strat.value] = results

    # All strategies should return results
    assert all(len(results) > 0 for results in results_by_strategy.values())

    # Different strategies may produce different rankings
    print("\nRanking strategy comparison:")
    for strat, results in results_by_strategy.items():
        print(f"\n{strat}:")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. Score: {r['final_score']:.3f}, Frequency: {r.get('frequency', 1)}")
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_multi_query_performance.py

import pytest
import time
import asyncio
from rag_factory.strategies.multi_query.strategy import MultiQueryRAGStrategy
from rag_factory.strategies.multi_query.config import MultiQueryConfig

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_variant_generation_performance(test_llm_service):
    """Benchmark variant generation speed."""
    from rag_factory.strategies.multi_query.variant_generator import QueryVariantGenerator

    config = MultiQueryConfig(num_variants=5)
    generator = QueryVariantGenerator(test_llm_service, config)

    query = "Explain machine learning algorithms"

    start = time.time()
    variants = await generator.generate_variants(query)
    duration = time.time() - start

    print(f"\nVariant generation: {len(variants)} variants in {duration:.3f}s")

    # Should be < 1 second
    assert duration < 1.0, f"Took {duration:.2f}s (expected <1s)"

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_parallel_execution_speedup(test_vector_store):
    """Benchmark parallel execution speedup vs sequential."""
    from rag_factory.strategies.multi_query.parallel_executor import ParallelQueryExecutor

    config = MultiQueryConfig(top_k_per_variant=10)
    executor = ParallelQueryExecutor(test_vector_store, config)

    variants = [f"query {i}" for i in range(5)]

    # Parallel execution
    start = time.time()
    results = await executor.execute_queries(variants)
    parallel_duration = time.time() - start

    # Sequential execution (for comparison)
    start = time.time()
    sequential_results = []
    for variant in variants:
        result = await executor._execute_single_query(variant, variant_index=0)
        sequential_results.append(result)
    sequential_duration = time.time() - start

    print(f"\nParallel: {parallel_duration:.3f}s")
    print(f"Sequential: {sequential_duration:.3f}s")
    print(f"Speedup: {sequential_duration / parallel_duration:.2f}x")

    # Parallel should be significantly faster
    assert parallel_duration < sequential_duration * 0.7

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_end_to_end_latency(test_vector_store, test_llm_service):
    """Benchmark complete end-to-end latency."""
    config = MultiQueryConfig(num_variants=5, final_top_k=5)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        config=config
    )

    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning",
        "What is supervised learning?",
        "Define reinforcement learning"
    ]

    latencies = []

    for query in queries:
        start = time.time()
        results = await strategy.aretrieve(query)
        duration = time.time() - start
        latencies.append(duration)

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\nEnd-to-end latency:")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Max: {max_latency:.3f}s")
    print(f"  Min: {min(latencies):.3f}s")

    # Average should be < 3 seconds
    assert avg_latency < 3.0, f"Average latency {avg_latency:.2f}s (expected <3s)"

@pytest.mark.benchmark
def test_throughput_queries_per_second(test_vector_store, test_llm_service):
    """Benchmark throughput (queries per second)."""
    config = MultiQueryConfig(num_variants=3, final_top_k=5)

    strategy = MultiQueryRAGStrategy(
        vector_store_service=test_vector_store,
        llm_service=test_llm_service,
        config=config
    )

    num_queries = 20
    queries = [f"Test query {i}" for i in range(num_queries)]

    start = time.time()

    # Process queries
    loop = asyncio.get_event_loop()
    for query in queries:
        loop.run_until_complete(strategy.aretrieve(query))

    duration = time.time() - start
    qps = num_queries / duration

    print(f"\nThroughput: {qps:.2f} queries/second")
    print(f"Total: {num_queries} queries in {duration:.2f}s")

    # Should process at least 3 queries per second
    assert qps >= 3.0, f"Throughput {qps:.2f} qps (expected >=3)"
```

---

## Definition of Done

- [ ] Query variant generator implemented with LLM integration
- [ ] Parallel query executor implemented with asyncio
- [ ] Result deduplicator working (exact + near-duplicate)
- [ ] At least 3 ranking strategies implemented (Max Score, RRF, Frequency Boost)
- [ ] Main MultiQueryRAGStrategy complete
- [ ] Configuration system working
- [ ] Async and sync retrieve methods working
- [ ] Fallback to original query on failures
- [ ] Error handling comprehensive
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements (<3s latency)
- [ ] Quality tests show improvement over baseline (>10% recall improvement)
- [ ] Logging and observability implemented
- [ ] Documentation complete with examples
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install aiohttp tenacity

# Verify async support
python -c "import asyncio; print('Asyncio available')"
```

### Configuration

```yaml
# config.yaml
strategies:
  multi_query:
    enabled: true

    # Variant generation
    num_variants: 3
    variant_types: ["paraphrase", "expand"]
    include_original: true

    # LLM settings
    llm_model: "gpt-3.5-turbo"
    llm_temperature: 0.7
    variant_generation_timeout: 5.0

    # Execution
    query_timeout: 10.0
    top_k_per_variant: 10

    # Deduplication
    enable_near_duplicate_detection: false
    near_duplicate_threshold: 0.95

    # Ranking
    ranking_strategy: "rrf"  # max_score, rrf, frequency_boost, hybrid
    frequency_boost_weight: 0.2
    rrf_k: 60

    # Output
    final_top_k: 5

    # Fallback
    fallback_to_original: true
    min_successful_queries: 1
```

### Usage Example

```python
from rag_factory.strategies.multi_query import MultiQueryRAGStrategy, MultiQueryConfig, RankingStrategy
import asyncio

# Setup strategy
config = MultiQueryConfig(
    num_variants=5,
    ranking_strategy=RankingStrategy.RECIPROCAL_RANK_FUSION,
    final_top_k=10
)

strategy = MultiQueryRAGStrategy(
    vector_store_service=vector_store,
    llm_service=llm,
    embedding_service=embedding_service,
    config=config
)

# Async usage (recommended)
async def retrieve():
    query = "What are the benefits of machine learning?"
    results = await strategy.aretrieve(query)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['final_score']:.3f}")
        print(f"   Found by {result['frequency']} variants")
        print(f"   Text: {result['text'][:100]}...")

asyncio.run(retrieve())

# Sync usage
results = strategy.retrieve("What is deep learning?")
```

---

## Notes for Developers

1. **Variant Quality**: The quality of generated variants depends heavily on the LLM and prompt. Tune the prompt template for your domain.

2. **Async Execution**: Always use async methods (`aretrieve`) for best performance. Sync wrapper is provided for compatibility but has overhead.

3. **Ranking Strategy Selection**:
   - **Max Score**: Simplest, works well when variants are similar
   - **RRF**: Best for diverse variants, resistant to outliers
   - **Frequency Boost**: Good when multiple variants finding same chunk indicates relevance
   - **Hybrid**: Combines RRF + frequency, usually best overall

4. **Performance Tuning**:
   - Reduce `num_variants` if latency is too high
   - Increase `top_k_per_variant` to improve recall
   - Enable `near_duplicate_detection` only if needed (adds overhead)

5. **Cost Optimization**: Each query variant costs an embedding computation. Monitor LLM API costs for variant generation.

6. **Fallback Strategy**: Always enable fallback to ensure system degrades gracefully.

7. **Testing**: Test with real LLM APIs to validate variant quality. Mock LLMs may not generate realistic variants.

8. **Monitoring**: Track variant diversity, result overlap, and ranking effectiveness in production.

9. **Error Handling**: Partial failures are acceptable. System should work even if some variants fail.

10. **Comparison with Baseline**: Always A/B test multi-query vs single-query to validate improvements for your use case.
