#!/usr/bin/env python3
"""Convert PyTorch embedding models to ONNX format.

This script converts sentence-transformers and HuggingFace models to ONNX format
for use with the ONNX local embedding provider.

Usage:
    python scripts/convert_model_to_onnx.py \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --output-dir ./onnx_models \\
        --quantize

Requirements:
    - optimum[onnxruntime]
    - transformers
    - torch (only for conversion, not for using ONNX models)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_factory.services.utils.model_converter import (
    convert_to_onnx,
    validate_conversion,
    get_model_size,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch embedding models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a model with default settings
  python scripts/convert_model_to_onnx.py \\
      --model-name sentence-transformers/all-MiniLM-L6-v2

  # Convert with quantization for smaller size
  python scripts/convert_model_to_onnx.py \\
      --model-name BAAI/bge-small-en-v1.5 \\
      --quantize

  # Convert to specific output directory
  python scripts/convert_model_to_onnx.py \\
      --model-name sentence-transformers/all-mpnet-base-v2 \\
      --output-dir /path/to/models

  # Skip validation (faster but not recommended)
  python scripts/convert_model_to_onnx.py \\
      --model-name my-custom-model \\
      --no-validate
        """
    )
    
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./onnx_models",
        help="Output directory for converted models (default: ./onnx_models)"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization to reduce model size"
    )
    
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip model optimization (not recommended)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation against original model (not recommended)"
    )
    
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity for validation (default: 0.99)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("ONNX Model Conversion")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quantize: {args.quantize}")
    logger.info(f"Optimize: {not args.no_optimize}")
    logger.info(f"Validate: {not args.no_validate}")
    logger.info("=" * 70)
    
    try:
        # Convert model
        logger.info("\n[1/3] Converting model to ONNX...")
        output_path = convert_to_onnx(
            model_name=args.model_name,
            output_dir=Path(args.output_dir),
            quantize=args.quantize,
            optimize=not args.no_optimize,
            opset_version=args.opset_version,
        )
        
        logger.info(f"✓ Conversion complete: {output_path}")
        
        # Get model size
        logger.info("\n[2/3] Analyzing model size...")
        size_info = get_model_size(output_path)
        logger.info(f"  Total size: {size_info['size_mb']:.2f} MB")
        logger.info(f"  Number of files: {size_info['num_files']}")
        
        # Validate conversion
        if not args.no_validate:
            logger.info("\n[3/3] Validating conversion...")
            validation_results = validate_conversion(
                original_model_name=args.model_name,
                onnx_model_path=output_path,
                similarity_threshold=args.similarity_threshold,
            )
            
            if validation_results["passed"]:
                logger.info("✓ Validation passed!")
                logger.info(f"  Mean similarity: {validation_results['mean_similarity']:.4f}")
                logger.info(f"  Min similarity: {validation_results['min_similarity']:.4f}")
            else:
                logger.warning("⚠ Validation warning!")
                logger.warning(f"  Mean similarity: {validation_results['mean_similarity']:.4f}")
                logger.warning(f"  Min similarity: {validation_results['min_similarity']:.4f}")
                logger.warning(f"  Threshold: {validation_results['threshold']}")
                logger.warning("\nThe model may still work, but embeddings differ from the original.")
        else:
            logger.info("\n[3/3] Skipping validation (--no-validate)")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✓ Model converted successfully")
        logger.info(f"  Location: {output_path}")
        logger.info(f"  Size: {size_info['size_mb']:.2f} MB")
        if not args.no_validate:
            logger.info(f"  Quality: {validation_results['mean_similarity']:.4f} similarity")
        logger.info("\nTo use this model:")
        logger.info(f"  from rag_factory.services.embeddings import ONNXLocalProvider")
        logger.info(f"  provider = ONNXLocalProvider({{'model_path': '{output_path}'}})")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Conversion failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Ensure PyTorch and transformers are installed:")
        logger.error("     pip install torch transformers optimum[onnxruntime]")
        logger.error("  2. Check that the model name is correct")
        logger.error("  3. Ensure you have internet connection for model download")
        logger.error("  4. Check the logs above for specific error details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
