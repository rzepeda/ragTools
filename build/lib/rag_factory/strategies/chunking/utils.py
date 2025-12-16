"""Utility functions for chunking strategies."""

from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Split on periods, exclamation marks, and question marks followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Text to split

    Returns:
        List of paragraphs
    """
    # Split by double newlines
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def is_code_block(text: str) -> bool:
    """Check if text is a code block.

    Args:
        text: Text to check

    Returns:
        True if code block detected
    """
    text_stripped = text.strip()

    # Check for fenced code blocks
    if text_stripped.startswith("```"):
        return True

    # Check for indented code blocks
    lines = text.split("\n")
    if len(lines) > 2:
        indented_lines = sum(1 for line in lines if line.startswith("    "))
        if indented_lines / len(lines) > 0.5:
            return True

    return False


def is_table(text: str) -> bool:
    """Check if text is a markdown table.

    Args:
        text: Text to check

    Returns:
        True if table detected
    """
    lines = text.split("\n")

    # Look for table rows (lines with multiple |)
    table_lines = [
        line for line in lines
        if "|" in line and line.count("|") >= 2
    ]

    return len(table_lines) >= 2


def extract_markdown_headers(text: str) -> List[Dict[str, Any]]:
    """Extract markdown headers from text.

    Args:
        text: Text to extract headers from

    Returns:
        List of header dictionaries with level, text, and position
    """
    headers = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            header_text = match.group(2).strip()
            headers.append({
                "level": level,
                "text": header_text,
                "line": i,
                "full_line": line
            })

    return headers


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)

    return text.strip()


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """Estimate reading time for text.

    Args:
        text: Text to estimate reading time for
        words_per_minute: Average reading speed

    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return word_count / words_per_minute


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def calculate_chunk_overlap_indices(
    chunks: List[str],
    overlap_size: int
) -> List[tuple]:
    """Calculate overlap indices for chunks.

    Args:
        chunks: List of chunk texts
        overlap_size: Overlap size in characters

    Returns:
        List of (start, end) tuples for each chunk with overlap
    """
    indices = []
    current_pos = 0

    for i, chunk in enumerate(chunks):
        chunk_length = len(chunk)

        if i == 0:
            # First chunk: no overlap at start
            start = 0
            end = chunk_length
        elif i == len(chunks) - 1:
            # Last chunk: overlap at start only
            start = max(0, current_pos - overlap_size)
            end = current_pos + chunk_length
        else:
            # Middle chunks: overlap on both sides
            start = max(0, current_pos - overlap_size)
            end = current_pos + chunk_length

        indices.append((start, end))
        current_pos += chunk_length

    return indices


def merge_small_chunks(
    chunks: List[str],
    min_size: int,
    max_size: int
) -> List[str]:
    """Merge chunks that are too small.

    Args:
        chunks: List of chunk texts
        min_size: Minimum chunk size in characters
        max_size: Maximum chunk size in characters

    Returns:
        List of merged chunks
    """
    if not chunks:
        return []

    merged = []
    current = chunks[0]

    for i in range(1, len(chunks)):
        # If current chunk is too small, try to merge
        if len(current) < min_size:
            # Check if merging would exceed max size
            if len(current) + len(chunks[i]) <= max_size:
                current = current + " " + chunks[i]
            else:
                # Can't merge, add current and start new
                merged.append(current)
                current = chunks[i]
        else:
            # Current chunk is good size, add it
            merged.append(current)
            current = chunks[i]

    # Add final chunk
    merged.append(current)

    return merged


def detect_language(text: str) -> str:
    """Detect programming language in code block.

    Simple heuristic-based detection.

    Args:
        text: Code text

    Returns:
        Detected language or 'unknown'
    """
    text_lower = text.lower()

    # Check for language indicators
    if 'def ' in text or 'import ' in text or 'print(' in text:
        return 'python'
    elif 'function ' in text or 'const ' in text or 'let ' in text:
        return 'javascript'
    elif 'public class' in text or 'private ' in text:
        return 'java'
    elif '#include' in text or 'int main' in text:
        return 'c'
    elif 'SELECT ' in text_lower or 'FROM ' in text_lower:
        return 'sql'

    return 'unknown'
