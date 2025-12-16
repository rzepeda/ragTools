"""ONNX Runtime utilities for embedding models.

This module provides utilities for working with ONNX models including:
- Model downloading from HuggingFace Hub
- ONNX Runtime session creation and optimization
- Model validation and metadata extraction
- Tokenizer loading and management
"""

from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING
from pathlib import Path
import logging
import json
import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False



logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available.
    
    Raises:
        ImportError: If required packages are not installed
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX Runtime not installed. Install with:\n"
            "  pip install onnx>=1.15.0 onnxruntime>=1.16.3\n"
            "Total size: ~215MB (vs ~2.5GB for PyTorch)"
        )
    


def get_onnx_model_path(
    model_name: str,
    cache_dir: Optional[Path] = None,
    filename: str = "model.onnx",
) -> Path:
    """Get path to local ONNX model.
    
    This function checks:
    1. Local model paths (if model exists locally)
    2. Local cache directory
    
    Args:
        model_name: Model identifier or local path
        cache_dir: Directory for model caching
        filename: Name of the ONNX model file
        
    Returns:
        Path to ONNX model file
        
    Raises:
        FileNotFoundError: If model is not found locally
    """
    import os
    
    check_dependencies()
    
    # Check for environment variable configuration
    env_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    env_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    
    # Use environment variable for cache_dir if not specified
    if cache_dir is None:
        if env_model_path:
            cache_dir = Path(env_model_path)
        elif Path("models/embeddings").exists():
            cache_dir = Path("models/embeddings").resolve()
            logger.info(f"Using local project cache: {cache_dir}")
        else:
            # Default to standard location, but don't create it if we're not downloading
            cache_dir = Path.home() / ".cache" / "rag_factory" / "onnx_models"
    
    cache_dir = Path(cache_dir)
    
    # Use environment variable for model_name if it matches default
    if env_model_name and model_name == "Xenova/all-MiniLM-L6-v2":
        logger.info(f"Using model from EMBEDDING_MODEL_NAME env: {env_model_name}")
        model_name = env_model_name
    
    # Check if model_name is a local path
    potential_local_path = Path(model_name)
    if potential_local_path.exists() and potential_local_path.is_file():
        logger.info(f"Using local ONNX model: {potential_local_path}")
        return potential_local_path
    
    # Check if model exists in cache directory
    # Try both naming conventions: underscore (old) and double-dash (HF standard)
    model_dir_name_underscore = model_name.replace("/", "_")
    model_dir_name_dash = model_name.replace("/", "--")
    
    for model_dir_name in [model_dir_name_dash, model_dir_name_underscore]:
        local_model_dir = cache_dir / model_dir_name
        
        if local_model_dir.exists():
            # Look for ONNX files in the directory (recursively, Xenova models have onnx/ subdirectory)
            onnx_files = list(local_model_dir.rglob("*.onnx"))
            if onnx_files:
                # Prefer the main model.onnx if available, otherwise use first one
                main_model = next((f for f in onnx_files if f.name == "model.onnx"), onnx_files[0])
                logger.info(f"Using cached ONNX model: {main_model}")
                return main_model
    
    # Model not found locally
    error_msg = (
        f"ONNX model '{model_name}' not found locally.\n"
        f"Checked path: {cache_dir}\n"
        f"Automatic downloading is disabled. Please ensure the model is available directly in the infrastructure.\n"
        f"Set EMBEDDING_MODEL_PATH environment variable to the directory containing the model."
    )
    raise FileNotFoundError(error_msg)


@DeprecationWarning
def download_tokenizer_config(
    model_name: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieves tokenizer configuration for local models.
    
    DEPRECATED: Downloading is disabled. This now implies looking for local config.
    
    Args:
        model_name: Model identifier
        cache_dir: Directory for caching
        
    Returns:
        Tokenizer configuration dictionary
    """
    check_dependencies()
    
    # Removed downloading logic
    logger.warning("download_tokenizer_config is deprecated and no longer downloads. Returning empty config.")
    return {}


def create_onnx_session(
    model_path: Path,
    providers: Optional[List[str]] = None,
    num_threads: Optional[int] = None,
    enable_profiling: bool = False,
    **session_options
) -> "ort.InferenceSession":
    """Create optimized ONNX Runtime inference session.
    
    Args:
        model_path: Path to ONNX model file
        providers: Execution providers (default: ['CPUExecutionProvider'])
        num_threads: Number of threads for CPU execution (default: auto)
        enable_profiling: Enable performance profiling
        **session_options: Additional ONNX session options
        
    Returns:
        Configured ONNX Runtime session
        
    Raises:
        ImportError: If ONNX Runtime is not installed
        ValueError: If model cannot be loaded
    """
    check_dependencies()
    
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    # Default to CPU provider (no GPU required)
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    # Create session options for optimization
    sess_options = ort.SessionOptions()
    
    # Enable all graph optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Set thread count if specified
    if num_threads is not None:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = max(1, num_threads // 2)
    
    # Enable profiling if requested
    if enable_profiling:
        sess_options.enable_profiling = True
    
    # Apply custom session options
    for key, value in session_options.items():
        if hasattr(sess_options, key):
            setattr(sess_options, key, value)
        else:
            logger.warning(f"Unknown session option: {key}")
    
    try:
        # Create session
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(
            f"Created ONNX session with providers: {session.get_providers()}"
        )
        return session
        
    except Exception as e:
        logger.error(f"Failed to create ONNX session: {e}")
        raise ValueError(f"Could not load ONNX model from {model_path}: {e}")


def validate_onnx_model(
    session: "ort.InferenceSession",
    expected_inputs: Optional[List[str]] = None,
    expected_outputs: Optional[List[str]] = None,
) -> bool:
    """Validate ONNX model structure.
    
    Args:
        session: ONNX Runtime session
        expected_inputs: Expected input names (optional)
        expected_outputs: Expected output names (optional)
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    # Get model inputs and outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    logger.info(f"Model has {len(inputs)} inputs and {len(outputs)} outputs")
    
    # Log input details
    for inp in inputs:
        logger.debug(f"  Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    
    # Log output details
    for out in outputs:
        logger.debug(f"  Output: {out.name}, shape: {out.shape}, type: {out.type}")
    
    # Validate expected inputs
    if expected_inputs:
        input_names = [inp.name for inp in inputs]
        for expected in expected_inputs:
            if expected not in input_names:
                raise ValueError(
                    f"Expected input '{expected}' not found in model. "
                    f"Available inputs: {input_names}"
                )
    
    # Validate expected outputs
    if expected_outputs:
        output_names = [out.name for out in outputs]
        for expected in expected_outputs:
            if expected not in output_names:
                raise ValueError(
                    f"Expected output '{expected}' not found in model. "
                    f"Available outputs: {output_names}"
                )
    
    logger.info("Model validation passed")
    return True


def get_model_metadata(session: "ort.InferenceSession") -> Dict[str, Any]:
    """Extract metadata from ONNX model.
    
    Args:
        session: ONNX Runtime session
        
    Returns:
        Dictionary with model metadata including:
        - input_names: List of input names
        - output_names: List of output names
        - input_shapes: Dictionary of input shapes
        - output_shapes: Dictionary of output shapes
        - embedding_dim: Embedding dimension (if detectable)
    """
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    metadata = {
        "input_names": [inp.name for inp in inputs],
        "output_names": [out.name for out in outputs],
        "input_shapes": {inp.name: inp.shape for inp in inputs},
        "output_shapes": {out.name: out.shape for out in outputs},
    }
    
    # Try to determine embedding dimension
    # Usually the last dimension of the first output
    if outputs:
        output_shape = outputs[0].shape
        if len(output_shape) >= 2:
            # Shape is typically [batch_size, sequence_length, embedding_dim]
            # or [batch_size, embedding_dim]
            embedding_dim = output_shape[-1]
            if isinstance(embedding_dim, int) and embedding_dim > 0:
                metadata["embedding_dim"] = embedding_dim
    
    return metadata


def mean_pooling(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray
) -> np.ndarray:
    """Apply mean pooling to token embeddings.
    
    This is the standard pooling strategy used by sentence-transformers.
    
    Args:
        token_embeddings: Token-level embeddings [batch, seq_len, dim]
        attention_mask: Attention mask [batch, seq_len]
        
    Returns:
        Pooled sentence embeddings [batch, dim]
    """
    # Expand attention mask to match embedding dimensions
    # [batch, seq_len] -> [batch, seq_len, 1]
    attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    
    # Convert to float for computation
    attention_mask_expanded = attention_mask_expanded.astype(np.float32)
    
    # Sum embeddings, weighted by attention mask
    sum_embeddings = np.sum(token_embeddings * attention_mask_expanded, axis=1)
    
    # Sum attention mask to get counts
    sum_mask = np.sum(attention_mask_expanded, axis=1)
    
    # Avoid division by zero
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    
    # Mean pooling
    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length (L2 normalization).
    
    Args:
        embeddings: Embedding vectors [batch, dim] or [dim]
        
    Returns:
        Normalized embeddings with unit length
    """
    # Handle both 1D and 2D arrays
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    # Compute L2 norms
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    
    # Normalize
    normalized = embeddings / norms
    
    if squeeze:
        normalized = normalized.squeeze(0)
    
    return normalized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    # Ensure vectors are 1D
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)
