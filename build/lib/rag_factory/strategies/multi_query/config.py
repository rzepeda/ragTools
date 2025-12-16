"""Configuration models for Multi-Query RAG Strategy."""

from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class VariantType(str, Enum):
    """Types of query variants to generate."""
    
    PARAPHRASE = "paraphrase"
    DECOMPOSE = "decompose"
    EXPAND = "expand"
    SPECIFY = "specify"
    GENERALIZE = "generalize"


class RankingStrategy(str, Enum):
    """Strategies for ranking merged results."""
    
    MAX_SCORE = "max_score"
    AVERAGE_SCORE = "average_score"
    WEIGHTED_AVERAGE = "weighted_average"
    FREQUENCY_BOOST = "frequency_boost"
    RECIPROCAL_RANK_FUSION = "rrf"
    HYBRID = "hybrid"


class MultiQueryConfig(BaseModel):
    """Configuration for multi-query RAG strategy.
    
    This configuration controls all aspects of the multi-query strategy:
    - Variant generation (how many, what types)
    - LLM settings for generation
    - Parallel execution parameters
    - Deduplication settings
    - Ranking strategy and parameters
    - Output settings
    - Fallback behavior
    - Observability options
    """

    # Variant generation
    num_variants: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of query variants to generate"
    )
    variant_types: List[VariantType] = Field(
        default=[VariantType.PARAPHRASE, VariantType.EXPAND],
        description="Types of variants to generate"
    )
    include_original: bool = Field(
        default=True,
        description="Include original query in variants"
    )

    # LLM settings
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for variant generation"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for variant generation"
    )
    variant_generation_timeout: float = Field(
        default=5.0,
        description="Timeout for variant generation (seconds)"
    )

    # Parallel execution
    query_timeout: float = Field(
        default=10.0,
        description="Timeout per query variant (seconds)"
    )
    max_concurrent_queries: int = Field(
        default=10,
        description="Max concurrent query executions"
    )

    # Results per variant
    top_k_per_variant: int = Field(
        default=10,
        description="Results to retrieve per variant"
    )

    # Deduplication
    enable_near_duplicate_detection: bool = Field(
        default=False,
        description="Enable embedding-based near-duplicate detection"
    )
    near_duplicate_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for near-duplicates"
    )

    # Ranking
    ranking_strategy: RankingStrategy = Field(
        default=RankingStrategy.RECIPROCAL_RANK_FUSION,
        description="Result ranking strategy"
    )
    frequency_boost_weight: float = Field(
        default=0.2,
        description="Weight for frequency boost in ranking"
    )
    rrf_k: int = Field(
        default=60,
        description="K parameter for Reciprocal Rank Fusion"
    )

    # Output
    final_top_k: int = Field(
        default=5,
        description="Final number of results to return"
    )

    # Fallback
    fallback_to_original: bool = Field(
        default=True,
        description="Fallback to original query if variant generation fails"
    )
    min_successful_queries: int = Field(
        default=1,
        description="Minimum successful queries required"
    )

    # Observability
    log_variants: bool = Field(
        default=True,
        description="Log generated variants"
    )
    track_metrics: bool = Field(
        default=True,
        description="Track performance metrics"
    )
