# Story 6.2: Implement Contextual Retrieval Strategy

**Story ID:** 6.2
**Epic:** Epic 6 - Multi-Query & Contextual Strategies
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 3 (Embedding Service, LLM Service), Epic 4 (Chunking Strategies)

---

## User Story

**As a** system
**I want** chunks enriched with document context
**So that** embeddings capture more contextual information and improve retrieval relevance

---

## Detailed Requirements

### Functional Requirements

1. **Context Generation for Chunks**
   - Use LLM to generate contextual descriptions for each chunk
   - Context should summarize:
     - Document/section the chunk belongs to
     - Topic/subject matter of the chunk
     - Relationship to surrounding content
     - Key entities or concepts in the chunk
   - Configurable context template/prompt
   - Context length configurable (50-200 tokens)
   - Preserve original chunk text unchanged

2. **Context Prepending to Chunks**
   - Prepend generated context to chunk text before embedding
   - Format: `[Context: {generated_context}]\n\n{original_chunk_text}`
   - Context marker clearly distinguishes context from content
   - Original chunk text remains unchanged in database
   - Store both contextualized and original versions

3. **Dual Storage System**
   - Store original chunk separately from contextualized version
   - Database schema:
     - `original_text`: Original chunk text
     - `context_description`: Generated context
     - `contextualized_text`: Context + original text
   - Embed contextualized_text for vector search
   - Return original_text (or both) in retrieval results
   - Metadata tracks context generation details

4. **Batch Processing for Efficiency**
   - Process multiple chunks in batches for LLM efficiency
   - Configurable batch size (10-50 chunks)
   - Parallel batch processing using async
   - Progress tracking for large document sets
   - Resume capability for interrupted processing
   - Batch retries on failures

5. **Context Generation Prompts**
   - Customizable prompt templates
   - Contextual information sources:
     - Document title/metadata
     - Section headers (hierarchy)
     - Preceding chunks (sliding window context)
     - Following chunks (look-ahead context)
     - Document-level summary (if available)
   - Domain-specific context templates
   - Support for different chunk types (text, code, table)

6. **Cost Tracking**
   - Track LLM API costs for context generation
   - Token usage monitoring:
     - Input tokens (chunk + prompt)
     - Output tokens (generated context)
     - Total cost per document/batch
   - Cost reporting and analytics
   - Budget limits and alerts
   - Cost optimization recommendations

7. **Selective Contextualization**
   - Option to contextualize only specific chunks:
     - Chunks below minimum size
     - Chunks lacking clear context
     - High-value chunks (detected by importance scoring)
   - Skip contextualization for:
     - Already contextual chunks (e.g., with headers)
     - Very short chunks (< threshold)
     - Code blocks (optional)
   - Configurable selection criteria

### Non-Functional Requirements

1. **Performance**
   - Context generation: <2s per batch (10-50 chunks)
   - Batch processing: >100 chunks/minute
   - Total document processing: <5min for 1000 chunks
   - Async/parallel processing for scalability
   - Efficient token usage (minimize prompt overhead)

2. **Cost Efficiency**
   - Minimize LLM API costs through:
     - Efficient prompt design
     - Batch processing
     - Caching where applicable
     - Selective contextualization
   - Target: <$0.01 per 1000 chunks (using GPT-3.5)
   - Configurable cost limits

3. **Quality**
   - Generated contexts should be:
     - Accurate and relevant
     - Concise (50-200 tokens)
     - Grammatically correct
     - Not redundant with chunk text
   - Improved retrieval accuracy vs baseline (>5% improvement)
   - Context quality validation

4. **Reliability**
   - Handle LLM API failures gracefully
   - Retry logic for transient failures
   - Fallback: Store chunks without context if generation fails
   - Partial success acceptable (some chunks without context)
   - Error logging and recovery

5. **Maintainability**
   - Clear separation of concerns (context generation, storage, retrieval)
   - Easy to update prompt templates
   - Support for multiple context generation strategies
   - Migration tools for adding context to existing chunks
   - Context regeneration capability

---

## Acceptance Criteria

### AC1: Context Generation
- [ ] LLM integration for context generation working
- [ ] Configurable context length (50-200 tokens)
- [ ] Customizable prompt templates
- [ ] Context includes document/section information
- [ ] Context is accurate and relevant
- [ ] Batch processing implemented (10-50 chunks per batch)

### AC2: Dual Storage
- [ ] Database schema supports dual storage
- [ ] Original text stored separately
- [ ] Context description stored
- [ ] Contextualized text stored
- [ ] Embeddings generated from contextualized text
- [ ] Retrieval returns original text (or both options)

### AC3: Contextual Information Sources
- [ ] Document metadata used in context generation
- [ ] Section hierarchy included in context
- [ ] Surrounding chunks considered (sliding window)
- [ ] Support for document-level summaries
- [ ] Different sources configurable

### AC4: Performance Requirements
- [ ] Batch processing <2s per batch (10-50 chunks)
- [ ] Throughput >100 chunks/minute
- [ ] Large documents (<5min for 1000 chunks)
- [ ] Async/parallel processing working
- [ ] No memory leaks

### AC5: Cost Tracking
- [ ] Token usage tracked (input + output)
- [ ] Cost calculated per batch/document
- [ ] Cost reporting available
- [ ] Budget limits enforceable
- [ ] Cost analytics dashboard/logs

### AC6: Selective Contextualization
- [ ] Option to contextualize only specific chunks
- [ ] Skip criteria implemented (short chunks, code blocks, etc.)
- [ ] Selection logic configurable
- [ ] Statistics on contextualization rate

### AC7: Quality Validation
- [ ] Generated contexts are relevant
- [ ] Improved retrieval accuracy (>5% vs baseline)
- [ ] Context quality metrics tracked
- [ ] A/B testing support

### AC8: Testing
- [ ] Unit tests for all components (>90% coverage)
- [ ] Integration tests with real LLM
- [ ] Performance benchmarks meet requirements
- [ ] Quality comparison tests vs baseline
- [ ] Cost tracking accuracy verified

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── contextual/
│   │   ├── __init__.py
│   │   ├── strategy.py              # Main contextual retrieval strategy
│   │   ├── context_generator.py     # LLM-based context generation
│   │   ├── batch_processor.py       # Batch processing for efficiency
│   │   ├── cost_tracker.py          # Track LLM API costs
│   │   ├── storage.py               # Dual storage management
│   │   ├── config.py                # Configuration models
│   │   └── prompts.py               # Context generation prompts

tests/
├── unit/
│   └── strategies/
│       └── contextual/
│           ├── test_context_generator.py
│           ├── test_batch_processor.py
│           ├── test_cost_tracker.py
│           └── test_storage.py
│
├── integration/
│   └── strategies/
│       └── test_contextual_integration.py
│
├── benchmarks/
│   └── test_contextual_performance.py
```

### Dependencies
```python
# requirements.txt additions
# No new dependencies - uses existing LLM and database services
```

### Database Schema Extension
```sql
-- Add context-related columns to chunks table
ALTER TABLE chunks ADD COLUMN context_description TEXT DEFAULT NULL;
ALTER TABLE chunks ADD COLUMN contextualized_text TEXT DEFAULT NULL;
ALTER TABLE chunks ADD COLUMN context_generation_method VARCHAR(50) DEFAULT NULL;
ALTER TABLE chunks ADD COLUMN context_token_count INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN context_cost DECIMAL(10, 6) DEFAULT 0.0;

-- Add index for querying contextualized chunks
CREATE INDEX idx_chunks_context_method ON chunks(context_generation_method);

-- Add table for context generation metadata
CREATE TABLE IF NOT EXISTS context_generation_log (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_length INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost DECIMAL(10, 6),
    llm_model VARCHAR(100),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX idx_context_log_chunk ON context_generation_log(chunk_id);
CREATE INDEX idx_context_log_document ON context_generation_log(document_id);
```

### Configuration Models
```python
# rag_factory/strategies/contextual/config.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ContextSource(Enum):
    """Sources of contextual information."""
    DOCUMENT_METADATA = "document_metadata"
    SECTION_HIERARCHY = "section_hierarchy"
    SURROUNDING_CHUNKS = "surrounding_chunks"
    DOCUMENT_SUMMARY = "document_summary"

class ContextGenerationMethod(Enum):
    """Methods for generating context."""
    LLM_FULL = "llm_full"              # Full LLM-generated context
    LLM_TEMPLATE = "llm_template"      # Template-based LLM context
    RULE_BASED = "rule_based"          # Rule-based context extraction
    HYBRID = "hybrid"                  # Combine LLM + rule-based

class ContextualRetrievalConfig(BaseModel):
    """Configuration for contextual retrieval strategy."""

    # Context generation
    enable_contextualization: bool = Field(default=True, description="Enable context generation")
    context_method: ContextGenerationMethod = Field(
        default=ContextGenerationMethod.LLM_TEMPLATE,
        description="Context generation method"
    )
    context_sources: List[ContextSource] = Field(
        default=[ContextSource.DOCUMENT_METADATA, ContextSource.SECTION_HIERARCHY],
        description="Sources for context generation"
    )

    # Context properties
    context_length_min: int = Field(default=50, description="Minimum context length (tokens)")
    context_length_max: int = Field(default=200, description="Maximum context length (tokens)")
    context_prefix: str = Field(default="Context:", description="Prefix for context in chunks")

    # LLM settings
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model for context generation")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Low temp for factual context")
    llm_max_tokens: int = Field(default=250, description="Max tokens for context generation")

    # Batch processing
    batch_size: int = Field(default=20, ge=1, le=100, description="Chunks per batch")
    enable_parallel_batches: bool = Field(default=True, description="Process batches in parallel")
    max_concurrent_batches: int = Field(default=5, description="Max concurrent batches")

    # Selective contextualization
    contextualize_all: bool = Field(default=True, description="Contextualize all chunks")
    min_chunk_size_for_context: int = Field(
        default=50,
        description="Minimum chunk size (tokens) to contextualize"
    )
    skip_code_blocks: bool = Field(default=False, description="Skip contextualization for code blocks")
    skip_tables: bool = Field(default=False, description="Skip contextualization for tables")

    # Storage
    store_original: bool = Field(default=True, description="Store original chunk text")
    store_context: bool = Field(default=True, description="Store generated context separately")
    store_contextualized: bool = Field(default=True, description="Store contextualized text")

    # Cost management
    enable_cost_tracking: bool = Field(default=True, description="Track LLM costs")
    cost_per_1k_input_tokens: float = Field(default=0.0015, description="Cost per 1K input tokens (USD)")
    cost_per_1k_output_tokens: float = Field(default=0.002, description="Cost per 1K output tokens (USD)")
    max_cost_per_document: Optional[float] = Field(default=None, description="Max cost per document (USD)")
    budget_alert_threshold: float = Field(default=10.0, description="Alert when cost exceeds threshold (USD)")

    # Error handling
    retry_on_failure: bool = Field(default=True, description="Retry on context generation failure")
    max_retries: int = Field(default=3, description="Max retries for failed chunks")
    fallback_to_no_context: bool = Field(default=True, description="Store chunk without context on failure")

    # Retrieval
    return_original_text: bool = Field(default=True, description="Return original text in results")
    return_context: bool = Field(default=False, description="Return context in results")
    return_contextualized: bool = Field(default=False, description="Return contextualized text in results")
```

### Context Generator
```python
# rag_factory/strategies/contextual/context_generator.py
from typing import List, Dict, Any, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import ContextualRetrievalConfig, ContextSource
from .prompts import CONTEXT_GENERATION_PROMPTS

logger = logging.getLogger(__name__)

class ContextGenerator:
    """Generates contextual descriptions for chunks using LLM."""

    def __init__(self, llm_service: Any, config: ContextualRetrievalConfig):
        """
        Initialize context generator.

        Args:
            llm_service: LLM service for generation
            config: Contextual retrieval configuration
        """
        self.llm_service = llm_service
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_context(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate contextual description for a chunk.

        Args:
            chunk: Chunk dict with text and metadata
            document_context: Optional document-level context information

        Returns:
            Generated context string, or None on failure
        """
        if not self.config.enable_contextualization:
            return None

        # Check if chunk should be contextualized
        if not self._should_contextualize(chunk):
            logger.debug(f"Skipping contextualization for chunk {chunk.get('chunk_id')}")
            return None

        try:
            # Build context generation prompt
            prompt = self._build_context_prompt(chunk, document_context)

            # Generate context using LLM
            response = await self.llm_service.agenerate(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )

            context = response.text.strip()

            # Validate context length
            context_tokens = self._count_tokens(context)
            if context_tokens < self.config.context_length_min:
                logger.warning(f"Generated context too short: {context_tokens} tokens")
                return None
            elif context_tokens > self.config.context_length_max:
                # Truncate if too long
                context = self._truncate_context(context, self.config.context_length_max)

            logger.debug(f"Generated context ({context_tokens} tokens): {context[:100]}...")

            return context

        except Exception as e:
            logger.error(f"Error generating context for chunk {chunk.get('chunk_id')}: {e}")

            if self.config.fallback_to_no_context:
                return None
            else:
                raise

    def _should_contextualize(self, chunk: Dict[str, Any]) -> bool:
        """Determine if chunk should be contextualized."""
        # Check contextualize_all setting
        if not self.config.contextualize_all:
            return False

        # Check minimum chunk size
        chunk_text = chunk.get("text", "")
        token_count = self._count_tokens(chunk_text)

        if token_count < self.config.min_chunk_size_for_context:
            logger.debug(f"Chunk too small ({token_count} tokens), skipping")
            return False

        # Check if code block (simple heuristic)
        if self.config.skip_code_blocks:
            if chunk_text.strip().startswith("```") or "def " in chunk_text or "class " in chunk_text:
                logger.debug("Detected code block, skipping")
                return False

        # Check if table
        if self.config.skip_tables:
            if "|" in chunk_text and chunk_text.count("|") > 5:
                logger.debug("Detected table, skipping")
                return False

        return True

    def _build_context_prompt(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for context generation."""
        chunk_text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # Gather contextual information based on sources
        context_parts = []

        if ContextSource.DOCUMENT_METADATA in self.config.context_sources:
            if document_context and "title" in document_context:
                context_parts.append(f"Document: {document_context['title']}")
            elif "document_id" in metadata:
                context_parts.append(f"Document ID: {metadata['document_id']}")

        if ContextSource.SECTION_HIERARCHY in self.config.context_sources:
            section_hierarchy = metadata.get("section_hierarchy", [])
            if section_hierarchy:
                context_parts.append(f"Section: {' > '.join(section_hierarchy)}")

        if ContextSource.SURROUNDING_CHUNKS in self.config.context_sources:
            # Would need preceding/following chunks from caller
            if document_context and "preceding_text" in document_context:
                context_parts.append(f"Preceding: {document_context['preceding_text'][:100]}...")

        # Build prompt
        contextual_info = "\n".join(context_parts) if context_parts else "No additional context available."

        prompt = f"""Generate a brief contextual description for the following text chunk.
The context should help understand what this chunk is about and where it fits in the larger document.

{contextual_info}

Chunk text:
{chunk_text}

Generate a concise context description (1-3 sentences, {self.config.context_length_min}-{self.config.context_length_max} tokens):"""

        return prompt

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (simplified - would use tiktoken in production)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to max tokens."""
        # Simple truncation - would use proper tokenization in production
        max_chars = max_tokens * 4
        if len(context) <= max_chars:
            return context

        # Truncate at sentence boundary if possible
        truncated = context[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:  # At least 70% of desired length
            return truncated[:last_period + 1]

        return truncated + "..."
```

### Batch Processor
```python
# rag_factory/strategies/contextual/batch_processor.py
import asyncio
from typing import List, Dict, Any, Optional
import logging
from .config import ContextualRetrievalConfig
from .context_generator import ContextGenerator
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes chunks in batches for efficient context generation."""

    def __init__(
        self,
        context_generator: ContextGenerator,
        cost_tracker: CostTracker,
        config: ContextualRetrievalConfig
    ):
        """
        Initialize batch processor.

        Args:
            context_generator: Context generator instance
            cost_tracker: Cost tracker instance
            config: Contextual retrieval configuration
        """
        self.context_generator = context_generator
        self.cost_tracker = cost_tracker
        self.config = config

    async def process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process chunks in batches to generate contexts.

        Args:
            chunks: List of chunks to process
            document_context: Optional document-level context

        Returns:
            List of chunks with generated contexts
        """
        logger.info(f"Processing {len(chunks)} chunks in batches of {self.config.batch_size}")

        # Split into batches
        batches = self._create_batches(chunks)

        # Process batches
        if self.config.enable_parallel_batches:
            processed_chunks = await self._process_batches_parallel(batches, document_context)
        else:
            processed_chunks = await self._process_batches_sequential(batches, document_context)

        logger.info(f"Processed {len(processed_chunks)} chunks")

        return processed_chunks

    def _create_batches(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split chunks into batches."""
        batches = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batches.append(batch)

        logger.info(f"Created {len(batches)} batches")
        return batches

    async def _process_batches_parallel(
        self,
        batches: List[List[Dict[str, Any]]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process batches in parallel."""
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_batch(batch, document_context)

        # Process all batches concurrently
        results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in batches],
            return_exceptions=True
        )

        # Flatten results
        processed_chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                continue
            processed_chunks.extend(result)

        return processed_chunks

    async def _process_batches_sequential(
        self,
        batches: List[List[Dict[str, Any]]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process batches sequentially."""
        processed_chunks = []

        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)}")

            try:
                batch_results = await self._process_batch(batch, document_context)
                processed_chunks.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                # Continue with next batch if fallback enabled
                if self.config.fallback_to_no_context:
                    processed_chunks.extend(batch)  # Add chunks without context
                else:
                    raise

        return processed_chunks

    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of chunks."""
        processed_chunks = []

        # Generate contexts for all chunks in batch concurrently
        tasks = [
            self._process_single_chunk(chunk, document_context)
            for chunk in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for chunk, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process chunk {chunk.get('chunk_id')}: {result}")

                if self.config.fallback_to_no_context:
                    processed_chunks.append(chunk)  # Add without context
                else:
                    raise result
            else:
                processed_chunks.append(result)

        return processed_chunks

    async def _process_single_chunk(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a single chunk."""
        chunk_id = chunk.get("chunk_id")

        try:
            # Generate context
            context = await self.context_generator.generate_context(chunk, document_context)

            if context:
                # Create contextualized text
                original_text = chunk.get("text", "")
                contextualized_text = f"{self.config.context_prefix} {context}\n\n{original_text}"

                # Track cost
                input_tokens = self.context_generator._count_tokens(original_text) + 100  # Approx prompt overhead
                output_tokens = self.context_generator._count_tokens(context)

                cost = self.cost_tracker.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                self.cost_tracker.record_chunk_cost(
                    chunk_id=chunk_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost
                )

                # Update chunk with context
                chunk["context_description"] = context
                chunk["contextualized_text"] = contextualized_text
                chunk["context_token_count"] = output_tokens
                chunk["context_cost"] = cost
                chunk["context_generation_method"] = self.config.context_method.value

                logger.debug(f"Chunk {chunk_id} contextualized (cost: ${cost:.6f})")

            return chunk

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            raise
```

### Cost Tracker
```python
# rag_factory/strategies/contextual/cost_tracker.py
from typing import Dict, Any
import logging
from .config import ContextualRetrievalConfig

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks LLM API costs for context generation."""

    def __init__(self, config: ContextualRetrievalConfig):
        """
        Initialize cost tracker.

        Args:
            config: Contextual retrieval configuration
        """
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.chunk_costs: Dict[str, Dict[str, Any]] = {}

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output_tokens

        return input_cost + output_cost

    def record_chunk_cost(
        self,
        chunk_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ) -> None:
        """
        Record cost for a chunk.

        Args:
            chunk_id: Chunk identifier
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            cost: Total cost
        """
        self.chunk_costs[chunk_id] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        # Check budget alert
        if (self.config.budget_alert_threshold and
            self.total_cost >= self.config.budget_alert_threshold):
            logger.warning(
                f"Cost alert: Total cost ${self.total_cost:.2f} "
                f"exceeds threshold ${self.config.budget_alert_threshold:.2f}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        avg_cost_per_chunk = (
            self.total_cost / len(self.chunk_costs)
            if self.chunk_costs else 0.0
        )

        return {
            "total_chunks": len(self.chunk_costs),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_chunk": avg_cost_per_chunk,
            "cost_per_1k_chunks": avg_cost_per_chunk * 1000 if avg_cost_per_chunk > 0 else 0
        }

    def reset(self) -> None:
        """Reset cost tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.chunk_costs = {}

    def check_budget_limit(self, max_cost: float) -> bool:
        """
        Check if cost is within budget.

        Args:
            max_cost: Maximum allowed cost

        Returns:
            True if within budget, False otherwise
        """
        return self.total_cost <= max_cost
```

### Storage Manager
```python
# rag_factory/strategies/contextual/storage.py
from typing import List, Dict, Any
import logging
from .config import ContextualRetrievalConfig

logger = logging.getLogger(__name__)

class ContextualStorageManager:
    """Manages storage of original and contextualized chunks."""

    def __init__(self, database_service: Any, config: ContextualRetrievalConfig):
        """
        Initialize storage manager.

        Args:
            database_service: Database service for chunk storage
            config: Contextual retrieval configuration
        """
        self.database = database_service
        self.config = config

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store chunks with dual storage (original + contextualized).

        Args:
            chunks: List of chunks with context information
        """
        logger.info(f"Storing {len(chunks)} contextualized chunks")

        for chunk in chunks:
            self._store_chunk(chunk)

    def _store_chunk(self, chunk: Dict[str, Any]) -> None:
        """Store a single chunk."""
        chunk_data = {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": chunk.get("document_id"),
        }

        # Store original text
        if self.config.store_original:
            chunk_data["original_text"] = chunk.get("text")

        # Store context
        if self.config.store_context and "context_description" in chunk:
            chunk_data["context_description"] = chunk.get("context_description")
            chunk_data["context_generation_method"] = chunk.get("context_generation_method")
            chunk_data["context_token_count"] = chunk.get("context_token_count", 0)
            chunk_data["context_cost"] = chunk.get("context_cost", 0.0)

        # Store contextualized text (for embedding)
        if self.config.store_contextualized and "contextualized_text" in chunk:
            chunk_data["contextualized_text"] = chunk.get("contextualized_text")
            chunk_data["text"] = chunk.get("contextualized_text")  # For embedding
        else:
            chunk_data["text"] = chunk.get("text")  # Use original

        # Store metadata
        chunk_data["metadata"] = chunk.get("metadata", {})

        # Save to database
        self.database.store_chunk(chunk_data)

    def retrieve_chunks(
        self,
        chunk_ids: List[str],
        return_format: str = "original"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with specified format.

        Args:
            chunk_ids: List of chunk IDs to retrieve
            return_format: Format to return ("original", "context", "contextualized", "both")

        Returns:
            List of chunks with requested format
        """
        chunks = self.database.get_chunks_by_ids(chunk_ids)

        # Format chunks based on return_format
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunk = chunk.copy()

            if return_format == "original":
                formatted_chunk["text"] = chunk.get("original_text", chunk.get("text"))
            elif return_format == "contextualized":
                formatted_chunk["text"] = chunk.get("contextualized_text", chunk.get("text"))
            elif return_format == "both":
                formatted_chunk["original_text"] = chunk.get("original_text")
                formatted_chunk["contextualized_text"] = chunk.get("contextualized_text")
                formatted_chunk["context"] = chunk.get("context_description")
            elif return_format == "context":
                formatted_chunk["context"] = chunk.get("context_description")

            formatted_chunks.append(formatted_chunk)

        return formatted_chunks
```

### Main Strategy Implementation
```python
# rag_factory/strategies/contextual/strategy.py
from typing import List, Dict, Any, Optional
import logging
import asyncio
from ..base import RAGStrategy
from .config import ContextualRetrievalConfig
from .context_generator import ContextGenerator
from .batch_processor import BatchProcessor
from .cost_tracker import CostTracker
from .storage import ContextualStorageManager

logger = logging.getLogger(__name__)

class ContextualRetrievalStrategy(RAGStrategy):
    """
    Contextual Retrieval: Enrich chunks with document context before embedding.

    This strategy:
    1. Generates contextual descriptions for each chunk using LLM
    2. Prepends context to chunk text before embedding
    3. Stores both original and contextualized versions
    4. Returns original text in retrieval results (configurable)
    """

    def __init__(
        self,
        vector_store_service: Any,
        database_service: Any,
        llm_service: Any,
        embedding_service: Any,
        config: Optional[ContextualRetrievalConfig] = None
    ):
        """
        Initialize contextual retrieval strategy.

        Args:
            vector_store_service: Vector store for retrieval
            database_service: Database for chunk storage
            llm_service: LLM service for context generation
            embedding_service: Embedding service for vectorization
            config: Contextual retrieval configuration
        """
        self.vector_store = vector_store_service
        self.database = database_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.config = config or ContextualRetrievalConfig()

        # Initialize components
        self.context_generator = ContextGenerator(llm_service, self.config)
        self.cost_tracker = CostTracker(self.config)
        self.batch_processor = BatchProcessor(
            self.context_generator,
            self.cost_tracker,
            self.config
        )
        self.storage_manager = ContextualStorageManager(database_service, self.config)

    async def aindex_document(
        self,
        document: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index document with contextual enrichment.

        Args:
            document: Original document text
            document_id: Document identifier
            chunks: Pre-chunked document chunks
            document_metadata: Optional document metadata

        Returns:
            Indexing statistics including cost information
        """
        logger.info(f"Indexing document with contextualization: {document_id}")

        # Reset cost tracker
        self.cost_tracker.reset()

        # Prepare document context
        document_context = {
            "document_id": document_id,
            "title": document_metadata.get("title") if document_metadata else document_id
        }

        # Process chunks in batches to generate contexts
        contextualized_chunks = await self.batch_processor.process_chunks(
            chunks,
            document_context
        )

        # Generate embeddings for contextualized text
        for chunk in contextualized_chunks:
            text_to_embed = chunk.get("contextualized_text") or chunk.get("text")

            embedding_result = self.embedding_service.embed([text_to_embed])
            chunk["embedding"] = embedding_result.embeddings[0]

        # Store chunks (dual storage)
        self.storage_manager.store_chunks(contextualized_chunks)

        # Index in vector store (using contextualized embeddings)
        for chunk in contextualized_chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk["chunk_id"],
                embedding=chunk["embedding"],
                metadata={
                    "document_id": document_id,
                    "has_context": "context_description" in chunk
                }
            )

        # Get cost summary
        cost_summary = self.cost_tracker.get_summary()

        logger.info(
            f"Indexed {len(contextualized_chunks)} chunks. "
            f"Cost: ${cost_summary['total_cost']:.4f}"
        )

        return {
            "document_id": document_id,
            "total_chunks": len(contextualized_chunks),
            "contextualized_chunks": sum(1 for c in contextualized_chunks if "context_description" in c),
            **cost_summary
        }

    def index_document(
        self,
        document: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous index document wrapper.

        Args:
            document: Original document text
            document_id: Document identifier
            chunks: Pre-chunked document chunks
            document_metadata: Optional document metadata

        Returns:
            Indexing statistics
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(
                self.aindex_document(document, document_id, chunks, document_metadata)
            )
        else:
            return loop.run_until_complete(
                self.aindex_document(document, document_id, chunks, document_metadata)
            )

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using contextualized embeddings.

        Args:
            query: User query
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            List of results (with original text by default)
        """
        logger.info(f"Contextual retrieval for: {query}")

        # Search using contextualized embeddings
        results = self.vector_store.search(query=query, top_k=top_k)

        # Get chunk IDs
        chunk_ids = [r.get("chunk_id") or r.get("id") for r in results]

        # Retrieve chunks with desired format
        return_format = "original" if self.config.return_original_text else "contextualized"

        if self.config.return_context:
            return_format = "both"

        formatted_chunks = self.storage_manager.retrieve_chunks(
            chunk_ids,
            return_format=return_format
        )

        # Merge with search results (scores, etc.)
        for result, chunk in zip(results, formatted_chunks):
            result.update(chunk)

        logger.info(f"Retrieved {len(results)} results")

        return results

    @property
    def name(self) -> str:
        return "contextual"

    @property
    def description(self) -> str:
        return "Enrich chunks with document context for improved retrieval"

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()
```

---

## Unit Tests

### Test Cases

#### TC6.2.1: Context Generator Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.contextual.context_generator import ContextGenerator
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

@pytest.fixture
def mock_llm_service():
    service = Mock()
    response = Mock()
    response.text = "This chunk discusses machine learning fundamentals in the context of an AI tutorial."
    service.agenerate = AsyncMock(return_value=response)
    return service

@pytest.fixture
def config():
    return ContextualRetrievalConfig(context_length_min=50, context_length_max=200)

@pytest.fixture
def context_generator(mock_llm_service, config):
    return ContextGenerator(mock_llm_service, config)

@pytest.mark.asyncio
async def test_generate_context_basic(context_generator, mock_llm_service):
    """Test basic context generation."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Machine learning is a subset of AI that enables systems to learn from data.",
        "metadata": {"document_id": "doc_1"}
    }

    context = await context_generator.generate_context(chunk)

    assert context is not None
    assert len(context) > 0
    mock_llm_service.agenerate.assert_called_once()

@pytest.mark.asyncio
async def test_context_includes_document_metadata(context_generator, mock_llm_service):
    """Test that context uses document metadata."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Test text",
        "metadata": {
            "document_id": "doc_1",
            "section_hierarchy": ["Chapter 1", "Section 1.1"]
        }
    }

    document_context = {"title": "AI Tutorial"}

    context = await context_generator.generate_context(chunk, document_context)

    # Check that prompt included metadata
    call_args = mock_llm_service.agenerate.call_args
    prompt = call_args[1]["prompt"]

    assert "AI Tutorial" in prompt or "Section" in prompt

@pytest.mark.asyncio
async def test_skip_short_chunks(context_generator):
    """Test that very short chunks are skipped."""
    chunk = {
        "chunk_id": "chunk_1",
        "text": "Short.",  # Very short
        "metadata": {}
    }

    context = await context_generator.generate_context(chunk)

    # Should skip due to length
    assert context is None

@pytest.mark.asyncio
async def test_skip_code_blocks(mock_llm_service):
    """Test skipping code blocks when configured."""
    config = ContextualRetrievalConfig(skip_code_blocks=True)
    generator = ContextGenerator(mock_llm_service, config)

    chunk = {
        "chunk_id": "chunk_1",
        "text": "```python\ndef hello():\n    print('Hello')\n```",
        "metadata": {}
    }

    context = await generator.generate_context(chunk)

    # Should skip code blocks
    assert context is None

@pytest.mark.asyncio
async def test_fallback_on_error(mock_llm_service):
    """Test fallback when context generation fails."""
    mock_llm_service.agenerate = AsyncMock(side_effect=Exception("LLM error"))

    config = ContextualRetrievalConfig(fallback_to_no_context=True)
    generator = ContextGenerator(mock_llm_service, config)

    chunk = {
        "chunk_id": "chunk_1",
        "text": "Some text here that should be long enough for contextualization.",
        "metadata": {}
    }

    context = await generator.generate_context(chunk)

    # Should return None (fallback)
    assert context is None

@pytest.mark.asyncio
async def test_context_length_validation(mock_llm_service):
    """Test context length validation and truncation."""
    # Mock very long response
    response = Mock()
    response.text = "This is a very long context. " * 100  # Very long
    mock_llm_service.agenerate = AsyncMock(return_value=response)

    config = ContextualRetrievalConfig(context_length_max=200)
    generator = ContextGenerator(mock_llm_service, config)

    chunk = {
        "chunk_id": "chunk_1",
        "text": "Test text that is long enough to generate context for it.",
        "metadata": {}
    }

    context = await generator.generate_context(chunk)

    # Should be truncated
    assert context is not None
    tokens = generator._count_tokens(context)
    assert tokens <= 250  # Allow some margin
```

#### TC6.2.2: Batch Processor Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.contextual.batch_processor import BatchProcessor
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

@pytest.fixture
def mock_context_generator():
    generator = Mock()

    async def mock_generate(chunk, doc_context=None):
        return f"Context for {chunk['chunk_id']}"

    generator.generate_context = mock_generate
    generator._count_tokens = lambda text: len(text) // 4
    return generator

@pytest.fixture
def mock_cost_tracker():
    tracker = Mock()
    tracker.calculate_cost.return_value = 0.001
    tracker.record_chunk_cost = Mock()
    return tracker

@pytest.fixture
def config():
    return ContextualRetrievalConfig(batch_size=5, enable_parallel_batches=True)

@pytest.fixture
def batch_processor(mock_context_generator, mock_cost_tracker, config):
    return BatchProcessor(mock_context_generator, mock_cost_tracker, config)

@pytest.mark.asyncio
async def test_process_chunks_batching(batch_processor):
    """Test that chunks are processed in batches."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(12)
    ]

    processed = await batch_processor.process_chunks(chunks)

    # All chunks should be processed
    assert len(processed) == 12
    assert all("context_description" in c for c in processed)

@pytest.mark.asyncio
async def test_parallel_batch_processing(batch_processor):
    """Test parallel batch processing."""
    import time

    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(20)
    ]

    start = time.time()
    processed = await batch_processor.process_chunks(chunks)
    duration = time.time() - start

    # Should process in parallel (faster than sequential)
    assert len(processed) == 20
    # Exact timing hard to test, but should be < sequential time

@pytest.mark.asyncio
async def test_cost_tracking_during_batch(batch_processor, mock_cost_tracker):
    """Test that costs are tracked during batch processing."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(5)
    ]

    await batch_processor.process_chunks(chunks)

    # Should record cost for each chunk
    assert mock_cost_tracker.record_chunk_cost.call_count == 5

@pytest.mark.asyncio
async def test_error_handling_in_batch(mock_context_generator, mock_cost_tracker):
    """Test error handling during batch processing."""
    call_count = 0

    async def flaky_generate(chunk, doc_context=None):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise Exception("Generation failed")
        return f"Context for {chunk['chunk_id']}"

    mock_context_generator.generate_context = flaky_generate

    config = ContextualRetrievalConfig(fallback_to_no_context=True, batch_size=5)
    processor = BatchProcessor(mock_context_generator, mock_cost_tracker, config)

    chunks = [{"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(5)]

    processed = await processor.process_chunks(chunks)

    # Should process all, some without context
    assert len(processed) == 5
    chunks_with_context = [c for c in processed if "context_description" in c]
    assert len(chunks_with_context) == 4  # One failed
```

#### TC6.2.3: Cost Tracker Tests
```python
import pytest
from rag_factory.strategies.contextual.cost_tracker import CostTracker
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

@pytest.fixture
def config():
    return ContextualRetrievalConfig(
        cost_per_1k_input_tokens=0.0015,
        cost_per_1k_output_tokens=0.002,
        budget_alert_threshold=1.0
    )

@pytest.fixture
def cost_tracker(config):
    return CostTracker(config)

def test_calculate_cost(cost_tracker):
    """Test cost calculation."""
    cost = cost_tracker.calculate_cost(input_tokens=1000, output_tokens=500)

    expected = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
    assert cost == pytest.approx(expected)

def test_record_chunk_cost(cost_tracker):
    """Test recording chunk cost."""
    cost_tracker.record_chunk_cost(
        chunk_id="chunk_1",
        input_tokens=500,
        output_tokens=100,
        cost=0.001
    )

    assert cost_tracker.total_input_tokens == 500
    assert cost_tracker.total_output_tokens == 100
    assert cost_tracker.total_cost == 0.001
    assert "chunk_1" in cost_tracker.chunk_costs

def test_cost_summary(cost_tracker):
    """Test cost summary generation."""
    # Record multiple chunks
    for i in range(10):
        cost_tracker.record_chunk_cost(
            chunk_id=f"chunk_{i}",
            input_tokens=100,
            output_tokens=50,
            cost=0.0005
        )

    summary = cost_tracker.get_summary()

    assert summary["total_chunks"] == 10
    assert summary["total_cost"] == 0.005
    assert summary["avg_cost_per_chunk"] == 0.0005

def test_budget_alert(cost_tracker, caplog):
    """Test budget alert threshold."""
    # Exceed budget threshold
    for i in range(100):
        cost_tracker.record_chunk_cost(
            chunk_id=f"chunk_{i}",
            input_tokens=1000,
            output_tokens=200,
            cost=0.02  # High cost
        )

    # Should trigger alert
    assert "Cost alert" in caplog.text

def test_reset(cost_tracker):
    """Test resetting cost tracker."""
    cost_tracker.record_chunk_cost("chunk_1", 100, 50, 0.001)

    cost_tracker.reset()

    assert cost_tracker.total_cost == 0.0
    assert len(cost_tracker.chunk_costs) == 0
```

---

## Integration Tests

### Test Scenarios

#### IS6.2.1: End-to-End Contextual Retrieval
```python
import pytest
from rag_factory.strategies.contextual.strategy import ContextualRetrievalStrategy
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_contextual_retrieval_complete_workflow(
    test_vector_store,
    test_database,
    test_llm_service,
    test_embedding_service
):
    """Test complete contextual retrieval workflow."""
    # Setup strategy
    config = ContextualRetrievalConfig(
        enable_contextualization=True,
        batch_size=10
    )

    strategy = ContextualRetrievalStrategy(
        vector_store_service=test_vector_store,
        database_service=test_database,
        llm_service=test_llm_service,
        embedding_service=test_embedding_service,
        config=config
    )

    # Prepare document chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "text": f"This is chunk {i} about machine learning concepts.",
            "metadata": {"section_hierarchy": ["Chapter 1", f"Section {i}"]}
        }
        for i in range(20)
    ]

    # Index document
    result = await strategy.aindex_document(
        document="Full document text",
        document_id="doc_1",
        chunks=chunks,
        document_metadata={"title": "ML Guide"}
    )

    # Check indexing result
    assert result["total_chunks"] == 20
    assert result["contextualized_chunks"] > 0
    assert result["total_cost"] > 0

    # Retrieve
    results = strategy.retrieve("machine learning", top_k=5)

    assert len(results) <= 5
    # Should return original text by default
    assert all("text" in r for r in results)

@pytest.mark.integration
def test_cost_tracking_accuracy(
    test_vector_store,
    test_database,
    test_llm_service,
    test_embedding_service
):
    """Test accuracy of cost tracking."""
    config = ContextualRetrievalConfig(enable_cost_tracking=True)

    strategy = ContextualRetrievalStrategy(
        vector_store_service=test_vector_store,
        database_service=test_database,
        llm_service=test_llm_service,
        embedding_service=test_embedding_service,
        config=config
    )

    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i} " * 20, "metadata": {}}
        for i in range(10)
    ]

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        strategy.aindex_document("doc", "doc_1", chunks)
    )

    # Verify cost tracking
    assert result["total_cost"] > 0
    assert result["total_input_tokens"] > 0
    assert result["total_output_tokens"] > 0

    cost_summary = strategy.get_cost_summary()
    assert cost_summary["total_cost"] == result["total_cost"]

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_quality_improvement(
    test_vector_store,
    test_database,
    test_llm_service,
    test_embedding_service
):
    """Test that contextualization improves retrieval quality."""
    # Test with contextualization
    config_with_context = ContextualRetrievalConfig(enable_contextualization=True)
    strategy_with = ContextualRetrievalStrategy(
        test_vector_store, test_database, test_llm_service,
        test_embedding_service, config_with_context
    )

    # Test without contextualization
    config_without_context = ContextualRetrievalConfig(enable_contextualization=False)
    strategy_without = ContextualRetrievalStrategy(
        test_vector_store, test_database, test_llm_service,
        test_embedding_service, config_without_context
    )

    # Index same chunks with both strategies
    test_chunks = [
        {"chunk_id": f"c{i}", "text": f"ML concept {i}", "metadata": {}}
        for i in range(50)
    ]

    await strategy_with.aindex_document("doc", "doc_w", test_chunks)
    await strategy_without.aindex_document("doc", "doc_wo", test_chunks)

    # Compare retrieval quality
    query = "machine learning concepts"
    results_with = strategy_with.retrieve(query, top_k=10)
    results_without = strategy_without.retrieve(query, top_k=10)

    # With context should have better or equal scores
    # (Exact comparison depends on test data quality)
    assert len(results_with) > 0
    assert len(results_without) > 0
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_contextual_performance.py

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_processing_throughput(test_llm_service):
    """Benchmark batch processing throughput."""
    from rag_factory.strategies.contextual.context_generator import ContextGenerator
    from rag_factory.strategies.contextual.batch_processor import BatchProcessor
    from rag_factory.strategies.contextual.cost_tracker import CostTracker
    from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

    config = ContextualRetrievalConfig(batch_size=20, enable_parallel_batches=True)
    generator = ContextGenerator(test_llm_service, config)
    tracker = CostTracker(config)
    processor = BatchProcessor(generator, tracker, config)

    # Generate large chunk set
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text content {i} " * 50, "metadata": {}}
        for i in range(200)
    ]

    import time
    start = time.time()
    processed = await processor.process_chunks(chunks)
    duration = time.time() - start

    chunks_per_minute = (len(processed) / duration) * 60

    print(f"\nBatch processing: {len(processed)} chunks in {duration:.2f}s")
    print(f"Throughput: {chunks_per_minute:.0f} chunks/minute")

    # Should meet >100 chunks/minute target
    assert chunks_per_minute >= 100, f"Throughput {chunks_per_minute:.0f} chunks/min (expected >=100)"
```

---

## Definition of Done

- [ ] Context generator implemented with LLM integration
- [ ] Batch processor working with parallel execution
- [ ] Cost tracker implemented and accurate
- [ ] Dual storage system working (original + contextualized)
- [ ] Context generation prompts customizable
- [ ] Selective contextualization working
- [ ] Main ContextualRetrievalStrategy complete
- [ ] Database schema migration complete
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements (>100 chunks/min)
- [ ] Cost tracking accurate and comprehensive
- [ ] Quality tests show improvement (>5% over baseline)
- [ ] Documentation complete with examples
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# No new dependencies required
# Uses existing LLM and database services
```

### Database Migration

```bash
# Apply context support migration
psql -d rag_database -f rag_factory/strategies/contextual/migrations/add_context_columns.sql

# Verify migration
psql -d rag_database -c "SELECT column_name FROM information_schema.columns WHERE table_name='chunks' AND column_name='context_description';"
```

### Configuration

```yaml
# config.yaml
strategies:
  contextual:
    enabled: true

    # Context generation
    enable_contextualization: true
    context_method: "llm_template"
    context_sources: ["document_metadata", "section_hierarchy"]
    context_length_min: 50
    context_length_max: 200

    # LLM settings
    llm_model: "gpt-3.5-turbo"
    llm_temperature: 0.3
    llm_max_tokens: 250

    # Batch processing
    batch_size: 20
    enable_parallel_batches: true
    max_concurrent_batches: 5

    # Selective contextualization
    contextualize_all: true
    min_chunk_size_for_context: 50
    skip_code_blocks: true
    skip_tables: false

    # Cost management
    enable_cost_tracking: true
    cost_per_1k_input_tokens: 0.0015
    cost_per_1k_output_tokens: 0.002
    budget_alert_threshold: 10.0

    # Retrieval
    return_original_text: true
    return_context: false
```

### Usage Example

```python
from rag_factory.strategies.contextual import ContextualRetrievalStrategy, ContextualRetrievalConfig
import asyncio

# Setup strategy
config = ContextualRetrievalConfig(
    enable_contextualization=True,
    batch_size=20,
    context_length_max=150
)

strategy = ContextualRetrievalStrategy(
    vector_store_service=vector_store,
    database_service=database,
    llm_service=llm,
    embedding_service=embedding_service,
    config=config
)

# Index document with contextualization
async def index():
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Content about topic {i}...",
            "metadata": {"section": f"Section {i}"}
        }
        for i in range(100)
    ]

    result = await strategy.aindex_document(
        document="Full document",
        document_id="doc_1",
        chunks=chunks,
        document_metadata={"title": "Technical Guide"}
    )

    print(f"Indexed {result['total_chunks']} chunks")
    print(f"Contextualized: {result['contextualized_chunks']}")
    print(f"Cost: ${result['total_cost']:.4f}")

asyncio.run(index())

# Retrieve
results = strategy.retrieve("technical information", top_k=5)

for result in results:
    print(f"\nChunk: {result['chunk_id']}")
    print(f"Text: {result['text'][:100]}...")
    if "context" in result:
        print(f"Context: {result['context']}")
```

---

## Notes for Developers

1. **Context Quality**: The quality of contexts directly impacts retrieval. Tune prompts for your domain.

2. **Cost Management**: Context generation adds LLM API costs. Monitor costs and use selective contextualization for large corpora.

3. **Batch Size Tuning**:
   - Larger batches: Better throughput, higher latency
   - Smaller batches: Lower latency, more API calls
   - Recommended: 10-20 chunks per batch

4. **Selective Contextualization**: Not all chunks benefit from context. Skip:
   - Very short chunks (already clear)
   - Code blocks (context may confuse)
   - Already contextual chunks (with section headers)

5. **Performance Optimization**:
   - Use parallel batch processing
   - Cache document-level contexts
   - Minimize prompt overhead

6. **Storage Considerations**: Dual storage increases storage by ~10-20%. Worth it for retrieval quality.

7. **Context vs Original Text**: Always return original text to users. Contextualized text is for embeddings only.

8. **Migration**: To add context to existing chunks, run batch reprocessing with contextualization enabled.

9. **Quality Validation**: A/B test contextualized vs non-contextualized retrieval to validate improvements.

10. **LLM Selection**: GPT-3.5-turbo is cost-effective for context generation. GPT-4 provides better quality but higher cost.
