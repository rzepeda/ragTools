"""
Configuration models for contextual retrieval strategy.

This module defines configuration classes and enums for the contextual
retrieval strategy, which enriches chunks with document context before embedding.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ContextSource(Enum):
    """Sources of contextual information for chunk enrichment."""
    
    DOCUMENT_METADATA = "document_metadata"
    SECTION_HIERARCHY = "section_hierarchy"
    SURROUNDING_CHUNKS = "surrounding_chunks"
    DOCUMENT_SUMMARY = "document_summary"


class ContextGenerationMethod(Enum):
    """Methods for generating contextual descriptions."""
    
    LLM_FULL = "llm_full"              # Full LLM-generated context
    LLM_TEMPLATE = "llm_template"      # Template-based LLM context
    RULE_BASED = "rule_based"          # Rule-based context extraction
    HYBRID = "hybrid"                  # Combine LLM + rule-based


class ContextualRetrievalConfig(BaseModel):
    """
    Configuration for contextual retrieval strategy.
    
    This configuration controls all aspects of contextual chunk enrichment,
    including context generation, batch processing, cost tracking, and storage.
    """

    # Context generation
    enable_contextualization: bool = Field(
        default=True,
        description="Enable context generation for chunks"
    )
    context_method: ContextGenerationMethod = Field(
        default=ContextGenerationMethod.LLM_TEMPLATE,
        description="Method for generating context"
    )
    context_sources: List[ContextSource] = Field(
        default=[ContextSource.DOCUMENT_METADATA, ContextSource.SECTION_HIERARCHY],
        description="Sources of contextual information to use"
    )

    # Context properties
    context_length_min: int = Field(
        default=50,
        ge=10,
        description="Minimum context length in tokens"
    )
    context_length_max: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Maximum context length in tokens"
    )
    context_prefix: str = Field(
        default="Context:",
        description="Prefix marker for context in chunks"
    )

    # LLM settings
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for context generation"
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for context generation (lower = more factual)"
    )
    llm_max_tokens: int = Field(
        default=250,
        ge=50,
        description="Maximum tokens for LLM context generation"
    )

    # Batch processing
    batch_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of chunks to process per batch"
    )
    enable_parallel_batches: bool = Field(
        default=True,
        description="Process batches in parallel for better throughput"
    )
    max_concurrent_batches: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent batches"
    )

    # Selective contextualization
    contextualize_all: bool = Field(
        default=True,
        description="Contextualize all chunks (if False, use selection criteria)"
    )
    min_chunk_size_for_context: int = Field(
        default=50,
        ge=10,
        description="Minimum chunk size in tokens to contextualize"
    )
    skip_code_blocks: bool = Field(
        default=False,
        description="Skip contextualization for code blocks"
    )
    skip_tables: bool = Field(
        default=False,
        description="Skip contextualization for tables"
    )

    # Storage
    store_original: bool = Field(
        default=True,
        description="Store original chunk text separately"
    )
    store_context: bool = Field(
        default=True,
        description="Store generated context description separately"
    )
    store_contextualized: bool = Field(
        default=True,
        description="Store contextualized text (context + original)"
    )

    # Cost management
    enable_cost_tracking: bool = Field(
        default=True,
        description="Track LLM API costs for context generation"
    )
    cost_per_1k_input_tokens: float = Field(
        default=0.0015,
        ge=0.0,
        description="Cost per 1000 input tokens in USD (GPT-3.5-turbo default)"
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.002,
        ge=0.0,
        description="Cost per 1000 output tokens in USD (GPT-3.5-turbo default)"
    )
    max_cost_per_document: Optional[float] = Field(
        default=None,
        description="Maximum cost per document in USD (None = no limit)"
    )
    budget_alert_threshold: float = Field(
        default=10.0,
        ge=0.0,
        description="Alert when total cost exceeds this threshold in USD"
    )

    # Error handling
    retry_on_failure: bool = Field(
        default=True,
        description="Retry context generation on failure"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed chunks"
    )
    fallback_to_no_context: bool = Field(
        default=True,
        description="Store chunk without context if generation fails"
    )

    # Retrieval
    return_original_text: bool = Field(
        default=True,
        description="Return original text in retrieval results"
    )
    return_context: bool = Field(
        default=False,
        description="Return context description in retrieval results"
    )
    return_contextualized: bool = Field(
        default=False,
        description="Return contextualized text in retrieval results"
    )

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
