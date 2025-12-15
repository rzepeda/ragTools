"""
Test utilities for embedding dimension handling.

This module provides utilities to get expected embedding dimensions
based on the model configured in the environment.
"""

import os
from typing import Dict


# Known model dimensions
KNOWN_MODEL_DIMENSIONS: Dict[str, int] = {
    "Xenova/all-MiniLM-L6-v2": 384,
    "Xenova/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


def get_expected_dimensions(model_name: str = None) -> int:
    """
    Get expected embedding dimensions for a model.
    
    Args:
        model_name: Model name. If None, uses EMBEDDING_MODEL_NAME from environment.
        
    Returns:
        Expected embedding dimensions (384 or 768)
        
    Raises:
        ValueError: If model is unknown
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")
    
    if model_name in KNOWN_MODEL_DIMENSIONS:
        return KNOWN_MODEL_DIMENSIONS[model_name]
    
    # Default based on common patterns
    if "MiniLM" in model_name or "small" in model_name.lower():
        return 384
    elif "mpnet" in model_name.lower() or "base" in model_name.lower():
        return 768
    
    # Default to 384 for unknown models (most common for fast models)
    return 384


def get_current_model_name() -> str:
    """
    Get the current embedding model name from environment.
    
    Returns:
        Model name from EMBEDDING_MODEL_NAME env var or default
    """
    return os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")


def get_current_dimensions() -> int:
    """
    Get expected dimensions for the current environment model.
    
    Returns:
        Expected embedding dimensions based on current environment
    """
    return get_expected_dimensions(get_current_model_name())
