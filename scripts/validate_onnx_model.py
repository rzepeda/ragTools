#!/usr/bin/env python3
"""Validate ONNX embedding models.

This script validates ONNX models by:
1. Checking model structure and metadata
2. Comparing embeddings with original PyTorch model
3. Running performance benchmarks
4. Generating a validation report

Usage:
    python scripts/validate_onnx_model.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --onnx-path ./onnx_models/sentence-transformers_all-MiniLM-L6-v2

    python scripts/validate_onnx_model.py \\
        --model BAAI/bge-small-en-v1.5 \\
        --threshold 0.99 \\
        --benchmark
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_factory.services.utils.onnx_utils import (
    create_onnx_session,
    validate_onnx_model,
    get_model_metadata,
)
from rag_factory.services.utils.model_converter import (
    validate_conversion,
    get_model_size,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_model(onnx_path: Path, num_iterations: int = 100) -> dict:
    """Benchmark ONNX model performance.
    
    Args:
        onnx_path: Path to ONNX model
        num_iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running performance benchmark ({num_iterations} iterations)...")
    
    # Create session
    session = create_onnx_session(onnx_path)
    
    # Test texts of varying lengths
    test_cases = {
        "short": "Test.",
        "medium": "This is a medium length test sentence for benchmarking." * 5,
        "long": "This is a longer test sentence that will be used for performance benchmarking. " * 20,
    }
    
    results = {}
    
    for name, text in test_cases.items():
        # Simple tokenization for benchmark
        words = text.split()[:512]
        input_ids = np.array([[hash(w) % 30000 for w in words] + [0] * (512 - len(words))], dtype=np.int64)
        attention_mask = np.array([[1] * len(words) + [0] * (512 - len(words))], dtype=np.int64)
        
        # Warm up
        for _ in range(10):
            session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        results[name] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
    
    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate ONNX embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Original HuggingFace model name for comparison"
    )
    
    parser.add_argument(
        "--onnx-path",
        help="Path to ONNX model directory (if not specified, will try to find it)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity threshold (default: 0.99)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking (default: 100)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("ONNX Model Validation")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 70)
    
    try:
        # Find ONNX model path
        if args.onnx_path:
            onnx_path = Path(args.onnx_path)
        else:
            # Try to find in default location
            model_dir_name = args.model.replace("/", "_")
            onnx_path = Path("./onnx_models") / model_dir_name
            
            if not onnx_path.exists():
                logger.error(f"ONNX model not found at {onnx_path}")
                logger.error("Please specify --onnx-path or convert the model first:")
                logger.error(f"  python scripts/convert_model_to_onnx.py --model-name {args.model}")
                return 1
        
        logger.info(f"ONNX model path: {onnx_path}")
        
        # Get model size
        logger.info("\n[1/4] Analyzing model size...")
        size_info = get_model_size(onnx_path)
        logger.info(f"  Size: {size_info['size_mb']:.2f} MB ({size_info['size_gb']:.3f} GB)")
        logger.info(f"  Files: {size_info['num_files']}")
        
        # Check model structure
        logger.info("\n[2/4] Validating model structure...")
        model_file = onnx_path / "model.onnx"
        if not model_file.exists():
            # Try to find any .onnx file
            onnx_files = list(onnx_path.glob("*.onnx"))
            if onnx_files:
                model_file = onnx_files[0]
            else:
                logger.error(f"No ONNX model file found in {onnx_path}")
                return 1
        
        session = create_onnx_session(model_file)
        validate_onnx_model(session, expected_inputs=["input_ids", "attention_mask"])
        
        metadata = get_model_metadata(session)
        logger.info(f"  Inputs: {metadata['input_names']}")
        logger.info(f"  Outputs: {metadata['output_names']}")
        if "embedding_dim" in metadata:
            logger.info(f"  Embedding dimension: {metadata['embedding_dim']}")
        
        # Validate against original model
        logger.info("\n[3/4] Comparing with original PyTorch model...")
        validation_results = validate_conversion(
            original_model_name=args.model,
            onnx_model_path=onnx_path,
            similarity_threshold=args.threshold,
        )
        
        if validation_results["passed"]:
            logger.info("✓ Quality validation passed!")
        else:
            logger.warning("⚠ Quality validation warning!")
        
        logger.info(f"  Mean similarity: {validation_results['mean_similarity']:.4f}")
        logger.info(f"  Min similarity: {validation_results['min_similarity']:.4f}")
        logger.info(f"  Threshold: {validation_results['threshold']}")
        
        # Run benchmarks
        if args.benchmark:
            logger.info(f"\n[4/4] Running performance benchmarks...")
            benchmark_results = benchmark_model(model_file, args.num_iterations)
            
            logger.info("\nPerformance Results:")
            logger.info("-" * 70)
            for text_type, stats in benchmark_results.items():
                logger.info(f"\n{text_type.capitalize()} text:")
                logger.info(f"  Mean: {stats['mean_ms']:.2f} ms")
                logger.info(f"  Std:  {stats['std_ms']:.2f} ms")
                logger.info(f"  P50:  {stats['p50_ms']:.2f} ms")
                logger.info(f"  P95:  {stats['p95_ms']:.2f} ms")
                logger.info(f"  P99:  {stats['p99_ms']:.2f} ms")
                
                # Check if meets performance target
                if stats['p95_ms'] < 100:
                    logger.info(f"  ✓ Meets <100ms target")
                else:
                    logger.warning(f"  ⚠ Exceeds 100ms target")
        else:
            logger.info("\n[4/4] Skipping benchmarks (use --benchmark to enable)")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Model: {args.model}")
        logger.info(f"Size: {size_info['size_mb']:.2f} MB")
        logger.info(f"Quality: {validation_results['mean_similarity']:.4f} similarity")
        
        if validation_results["passed"]:
            logger.info("Status: ✓ PASSED")
        else:
            logger.info("Status: ⚠ WARNING - Quality below threshold")
        
        if args.benchmark:
            logger.info(f"Performance: {benchmark_results['medium']['p95_ms']:.2f} ms (p95)")
        
        logger.info("=" * 70)
        
        return 0 if validation_results["passed"] else 1
        
    except Exception as e:
        logger.error(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
