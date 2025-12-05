"""
Example demonstrating query expansion functionality.

This script shows how to use the query expansion service with different strategies.
"""

import os
from rag_factory.strategies.query_expansion import (
    QueryExpanderService,
    ExpansionConfig,
    ExpansionStrategy
)
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig


def print_expansion_result(result, title="Query Expansion Result"):
    """Print expansion result in a readable format."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Original Query: {result.original_query}")
    print(f"Expanded Query: {result.primary_expansion.expanded_query}")
    print(f"Added Terms: {result.primary_expansion.added_terms}")
    print(f"Execution Time: {result.execution_time_ms:.0f}ms")
    print(f"Cache Hit: {result.cache_hit}")
    print(f"Strategy: {result.primary_expansion.expansion_strategy.value}")
    if result.primary_expansion.reasoning:
        print(f"Reasoning: {result.primary_expansion.reasoning}")
    print(f"{'='*60}")


def example_keyword_expansion():
    """Example: Keyword expansion strategy."""
    print("\n\n## Example 1: Keyword Expansion ##")

    # Configure LLM service
    llm_config = LLMServiceConfig(
        provider="openai",  # or "anthropic" or "ollama"
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    # Configure expansion
    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        max_additional_terms=5,
        enable_cache=True,
        track_metrics=True
    )

    # Create expander service
    service = QueryExpanderService(expansion_config, llm_service)

    # Expand query
    result = service.expand("machine learning")
    print_expansion_result(result, "Keyword Expansion")

    # Expand again to show cache hit
    result2 = service.expand("machine learning")
    print_expansion_result(result2, "Keyword Expansion (Cached)")

    # Show stats
    stats = service.get_stats()
    print(f"\nService Statistics:")
    print(f"  Total expansions: {stats['total_expansions']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")


def example_query_reformulation():
    """Example: Query reformulation strategy."""
    print("\n\n## Example 2: Query Reformulation ##")

    llm_config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.REFORMULATION,
        enable_cache=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    # Reformulate a vague query
    result = service.expand("how does it work")
    print_expansion_result(result, "Query Reformulation")


def example_hyde_expansion():
    """Example: HyDE (Hypothetical Document Expansion)."""
    print("\n\n## Example 3: HyDE Expansion ##")

    llm_config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.HYDE,
        max_tokens=150,
        enable_cache=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    # Generate hypothetical document
    result = service.expand("What is the capital of France?")
    print_expansion_result(result, "HyDE Expansion")


def example_multi_query():
    """Example: Multi-query generation."""
    print("\n\n## Example 4: Multi-Query Generation ##")

    llm_config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.MULTI_QUERY,
        generate_multiple_variants=True,
        num_variants=3,
        enable_cache=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    # Generate multiple query variants
    result = service.expand("climate change effects")

    print(f"\n{'='*60}")
    print("Multi-Query Generation")
    print(f"{'='*60}")
    print(f"Original Query: {result.original_query}")
    print(f"\nGenerated Variants ({len(result.expanded_queries)}):")
    for i, variant in enumerate(result.expanded_queries, 1):
        print(f"  {i}. {variant.expanded_query}")
    print(f"\nExecution Time: {result.execution_time_ms:.0f}ms")
    print(f"{'='*60}")


def example_ab_testing():
    """Example: A/B testing expansion vs no expansion."""
    print("\n\n## Example 5: A/B Testing ##")

    llm_config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        enable_cache=False,  # Disable cache for this demo
        track_metrics=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    query = "neural networks"

    # Test with expansion enabled
    result_expanded = service.expand(query, enable_expansion=True)
    print_expansion_result(result_expanded, "With Expansion")

    # Test with expansion disabled
    result_original = service.expand(query, enable_expansion=False)
    print_expansion_result(result_original, "Without Expansion")

    # Show A/B test stats
    stats = service.get_stats()
    print(f"\nA/B Test Statistics:")
    print(f"  Total requests: {stats['total_expansions']}")
    print(f"  Expanded: {stats['expansion_enabled_count']}")
    print(f"  Original: {stats['expansion_disabled_count']}")
    print(f"  Expansion rate: {stats['expansion_rate']:.2%}")


def example_domain_context():
    """Example: Using domain context for specialized expansion."""
    print("\n\n## Example 6: Domain Context ##")

    llm_config = LLMServiceConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        provider_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    # Medical domain
    expansion_config = ExpansionConfig(
        strategy=ExpansionStrategy.KEYWORD,
        domain_context="medical and healthcare context",
        enable_cache=True
    )

    service = QueryExpanderService(expansion_config, llm_service)

    result = service.expand("diagnosis")
    print_expansion_result(result, "Medical Domain Expansion")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: No API key found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        return

    print("="*60)
    print("Query Expansion Examples")
    print("="*60)

    try:
        # Run examples
        example_keyword_expansion()
        example_query_reformulation()
        example_hyde_expansion()
        example_multi_query()
        example_ab_testing()
        example_domain_context()

        print("\n\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have a valid API key set.")


if __name__ == "__main__":
    main()
