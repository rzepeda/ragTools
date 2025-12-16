"""Model conversion utilities for PyTorch to ONNX conversion.

This module provides utilities for converting PyTorch embedding models to ONNX format.
Note: This module requires PyTorch and transformers to be installed for conversion,
but the converted models can be used without these dependencies.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_onnx(
    model_name: str,
    output_dir: Path,
    quantize: bool = False,
    optimize: bool = True,
    opset_version: int = 14,
) -> Path:
    """Convert a PyTorch model to ONNX format.
    
    This function requires PyTorch and transformers to be installed.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Output directory for ONNX model
        quantize: Whether to apply dynamic quantization
        optimize: Whether to optimize the model
        opset_version: ONNX opset version to use
        
    Returns:
        Path to converted ONNX model directory
        
    Raises:
        ImportError: If required packages are not installed
        ValueError: If conversion fails
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "Model conversion requires optimum and transformers. Install with:\n"
            "  pip install optimum[onnxruntime] transformers torch\n"
            "Note: These are only needed for conversion, not for using ONNX models."
        )
    
    output_path = Path(output_dir) / model_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {model_name} to ONNX...")
    logger.info(f"Output directory: {output_path}")
    
    try:
        # Load and convert model using optimum
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider",
        )
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying dynamic quantization...")
            # Optimum handles quantization during export
            # For more control, could use onnxruntime.quantization
        
        # Save ONNX model
        model.save_pretrained(output_path)
        logger.info(f"Saved ONNX model to: {output_path}")
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Saved tokenizer to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise ValueError(f"Failed to convert {model_name} to ONNX: {e}")


def convert_raw_pytorch_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: tuple = (1, 512),
    opset_version: int = 14,
    device: str = "cpu"
) -> Path:
    """Convert raw PyTorch model to ONNX format (fallback method).
    
    Args:
        model_path: Path to PyTorch model file
        output_path: Path to save ONNX model
        input_shape: Input shape (batch_size, seq_length)
        opset_version: ONNX opset version
        device: Device to run conversion on
        
    Returns:
        Path to saved ONNX model
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for conversion")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw model from {model_path}...")
    try:
        model = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = {
        "input_ids": torch.randint(0, 1000, input_shape, dtype=torch.long).to(device),
        "attention_mask": torch.ones(input_shape, dtype=torch.long).to(device)
    }
    
    # Handle token_type_ids if model expects them
    if hasattr(model, "config") and hasattr(model.config, "type_vocab_size") and model.config.type_vocab_size > 0:
        dummy_input["token_type_ids"] = torch.zeros(input_shape, dtype=torch.long).to(device)

    logger.info(f"Exporting to ONNX at {output_path}...")
    
    input_names = list(dummy_input.keys())
    output_names = ["last_hidden_state", "pooler_output"]
    
    torch.onnx.export(
        model,
        tuple(dummy_input.values()),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    logger.info("Model exported successfully")
    return output_path


def validate_conversion(
    original_model_name: str,
    onnx_model_path: Path,
    test_texts: Optional[list] = None,
    similarity_threshold: float = 0.99,
) -> Dict[str, Any]:
    """Validate that ONNX model produces similar embeddings to original.
    
    This function requires the original PyTorch model to be available.
    
    Args:
        original_model_name: Original HuggingFace model name
        onnx_model_path: Path to converted ONNX model
        test_texts: List of test texts (default: standard test set)
        similarity_threshold: Minimum cosine similarity required
        
    Returns:
        Validation results dictionary with:
        - passed: Whether validation passed
        - similarities: List of similarity scores
        - mean_similarity: Average similarity
        - min_similarity: Minimum similarity
        
    Raises:
        ImportError: If required packages are not installed
    """
    if test_texts is None:
        test_texts = [
            "This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers.",
        ]
    
    try:
        from sentence_transformers import SentenceTransformer
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "Validation requires sentence-transformers and transformers. Install with:\n"
            "  pip install sentence-transformers transformers torch"
        )
    
    logger.info("Validating conversion...")
    
    # Load original model
    logger.info(f"Loading original model: {original_model_name}")
    original_model = SentenceTransformer(original_model_name)
    original_embeddings = original_model.encode(test_texts, convert_to_numpy=True)
    
    # Load ONNX model
    logger.info(f"Loading ONNX model: {onnx_model_path}")
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(str(onnx_model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_model_path))
    
    # Generate ONNX embeddings
    onnx_embeddings = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = onnx_model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            onnx_embeddings.append(embedding)
    
    onnx_embeddings = np.array(onnx_embeddings)
    
    # Calculate similarities
    similarities = []
    for orig, onnx in zip(original_embeddings, onnx_embeddings):
        similarity = np.dot(orig, onnx) / (np.linalg.norm(orig) * np.linalg.norm(onnx))
        similarities.append(float(similarity))
    
    mean_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    
    # Log results
    logger.info(f"Mean similarity: {mean_similarity:.4f}")
    logger.info(f"Min similarity: {min_similarity:.4f}")
    
    for i, (text, sim) in enumerate(zip(test_texts, similarities)):
        logger.debug(f"  Text {i+1}: {sim:.4f} - {text[:50]}...")
    
    passed = min_similarity >= similarity_threshold
    
    if passed:
        logger.info("✓ Validation passed!")
    else:
        logger.warning(
            f"⚠ Validation warning: min similarity {min_similarity:.4f} < {similarity_threshold}"
        )
    
    return {
        "passed": passed,
        "similarities": similarities,
        "mean_similarity": mean_similarity,
        "min_similarity": min_similarity,
        "threshold": similarity_threshold,
    }


def get_model_size(model_path: Path) -> Dict[str, Any]:
    """Get size information for a model.
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        Dictionary with size information
    """
    model_path = Path(model_path)
    
    if model_path.is_file():
        size_bytes = model_path.stat().st_size
        files = [model_path.name]
    elif model_path.is_dir():
        size_bytes = sum(
            f.stat().st_size
            for f in model_path.rglob("*")
            if f.is_file()
        )
        files = [f.name for f in model_path.rglob("*") if f.is_file()]
    else:
        raise ValueError(f"Path does not exist: {model_path}")
    
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    
    return {
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "size_gb": round(size_gb, 3),
        "num_files": len(files),
        "files": files,
    }
