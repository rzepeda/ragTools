"""
Prompt templates for context generation.

This module provides customizable prompt templates for generating
contextual descriptions of document chunks.
"""

from typing import Dict, Any, Optional


# Default context generation prompt template
DEFAULT_CONTEXT_PROMPT = """Generate a brief contextual description for the following text chunk.
The context should help understand what this chunk is about and where it fits in the larger document.

{contextual_info}

Chunk text:
{chunk_text}

Generate a concise context description (1-3 sentences, {min_tokens}-{max_tokens} tokens):"""


# Template for code chunks
CODE_CONTEXT_PROMPT = """Generate a brief contextual description for the following code chunk.
The context should explain what this code does and its purpose in the larger codebase.

{contextual_info}

Code chunk:
{chunk_text}

Generate a concise context description (1-3 sentences, {min_tokens}-{max_tokens} tokens):"""


# Template for table chunks
TABLE_CONTEXT_PROMPT = """Generate a brief contextual description for the following table.
The context should explain what data the table contains and its purpose.

{contextual_info}

Table:
{chunk_text}

Generate a concise context description (1-3 sentences, {min_tokens}-{max_tokens} tokens):"""


# Template for technical documentation
TECHNICAL_DOC_PROMPT = """Generate a brief contextual description for the following technical documentation chunk.
The context should summarize the technical topic and its relevance.

{contextual_info}

Documentation chunk:
{chunk_text}

Generate a concise technical context (1-3 sentences, {min_tokens}-{max_tokens} tokens):"""


CONTEXT_GENERATION_PROMPTS: Dict[str, str] = {
    "default": DEFAULT_CONTEXT_PROMPT,
    "code": CODE_CONTEXT_PROMPT,
    "table": TABLE_CONTEXT_PROMPT,
    "technical": TECHNICAL_DOC_PROMPT,
}


def build_contextual_info(
    document_title: Optional[str] = None,
    section_hierarchy: Optional[list] = None,
    preceding_text: Optional[str] = None,
    document_id: Optional[str] = None,
) -> str:
    """
    Build contextual information string from available sources.
    
    Args:
        document_title: Title of the document
        section_hierarchy: List of section headers (e.g., ["Chapter 1", "Section 1.1"])
        preceding_text: Text from preceding chunks
        document_id: Document identifier
        
    Returns:
        Formatted contextual information string
    """
    parts = []
    
    if document_title:
        parts.append(f"Document: {document_title}")
    elif document_id:
        parts.append(f"Document ID: {document_id}")
    
    if section_hierarchy:
        section_path = " > ".join(section_hierarchy)
        parts.append(f"Section: {section_path}")
    
    if preceding_text:
        # Truncate preceding text if too long
        max_preceding = 100
        if len(preceding_text) > max_preceding:
            preceding_text = preceding_text[:max_preceding] + "..."
        parts.append(f"Preceding context: {preceding_text}")
    
    if parts:
        return "\n".join(parts)
    else:
        return "No additional context available."


def get_prompt_template(chunk_type: str = "default") -> str:
    """
    Get prompt template for specific chunk type.
    
    Args:
        chunk_type: Type of chunk ("default", "code", "table", "technical")
        
    Returns:
        Prompt template string
    """
    return CONTEXT_GENERATION_PROMPTS.get(chunk_type, DEFAULT_CONTEXT_PROMPT)
