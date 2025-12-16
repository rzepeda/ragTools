"""Prompt templates for Multi-Query RAG Strategy."""

from typing import Dict
from .config import VariantType


# Default variant generation prompt
DEFAULT_VARIANT_PROMPT = """Generate {num_variants} diverse variants of the following query.
Each variant should capture the same intent but use different phrasings, perspectives, or levels of specificity.

Generate these types of variants: {variant_types}

Original Query: {query}

Generate exactly {num_variants} variants, one per line:"""


# Type-specific prompt templates
VARIANT_TYPE_PROMPTS: Dict[VariantType, str] = {
    VariantType.PARAPHRASE: """Generate {num_variants} paraphrased versions of the query below.
Each paraphrase should use different words and sentence structures while maintaining the exact same meaning.

Original Query: {query}

Paraphrased variants (one per line):""",
    
    VariantType.DECOMPOSE: """Break down the following complex query into {num_variants} simpler sub-queries.
Each sub-query should address a specific aspect of the original question.

Original Query: {query}

Sub-queries (one per line):""",
    
    VariantType.EXPAND: """Generate {num_variants} expanded versions of the query below.
Each expansion should add related concepts, context, or clarifying details.

Original Query: {query}

Expanded variants (one per line):""",
    
    VariantType.SPECIFY: """Generate {num_variants} more specific versions of the query below.
Each variant should narrow the focus or add specific constraints.

Original Query: {query}

More specific variants (one per line):""",
    
    VariantType.GENERALIZE: """Generate {num_variants} more general versions of the query below.
Each variant should broaden the scope or remove specific constraints.

Original Query: {query}

More general variants (one per line):"""
}


def get_variant_prompt(
    query: str,
    num_variants: int,
    variant_types: list,
    use_type_specific: bool = False
) -> str:
    """Get the appropriate prompt for variant generation.
    
    Args:
        query: Original user query
        num_variants: Number of variants to generate
        variant_types: List of VariantType enums
        use_type_specific: Whether to use type-specific prompts
        
    Returns:
        Formatted prompt string
    """
    if use_type_specific and len(variant_types) == 1:
        # Use type-specific prompt if only one type requested
        template = VARIANT_TYPE_PROMPTS.get(
            variant_types[0],
            DEFAULT_VARIANT_PROMPT
        )
        return template.format(query=query, num_variants=num_variants)
    else:
        # Use default prompt with variant types listed
        variant_types_str = ", ".join([vt.value for vt in variant_types])
        return DEFAULT_VARIANT_PROMPT.format(
            query=query,
            num_variants=num_variants,
            variant_types=variant_types_str
        )
