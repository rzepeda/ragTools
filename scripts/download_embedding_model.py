#!/usr/bin/env python3
"""Download ONNX embedding models from HuggingFace.

This script downloads pre-converted ONNX models from Xenova's repository
and saves them to the local models directory.

Usage:
    python scripts/download_embedding_model.py
    python scripts/download_embedding_model.py --model Xenova/all-MiniLM-L6-v2
    python scripts/download_embedding_model.py --output-dir custom/path
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available Xenova ONNX models
AVAILABLE_MODELS = {
    "Xenova/all-mpnet-base-v2": {
        "dimensions": 768,
        "description": "High quality, 768 dimensions (recommended)",
        "size_mb": 420
    },
    "Xenova/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Fast and lightweight, 384 dimensions",
        "size_mb": 90
    },
    "Xenova/paraphrase-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Optimized for paraphrase detection",
        "size_mb": 90
    },
}


def download_model(model_name: str, output_dir: Path) -> bool:
    """Download ONNX model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'Xenova/all-mpnet-base-v2')
        output_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    if not HF_AVAILABLE:
        logger.error("huggingface-hub not installed. Install with:")
        logger.error("  pip install huggingface-hub")
        return False
    
    try:
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model files
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(output_dir),
            local_dir=str(output_dir / model_name.replace("/", "_")),
            local_dir_use_symlinks=False,
        )
        
        logger.info(f"✅ Model downloaded successfully to: {model_path}")
        
        # Verify ONNX files exist (check recursively, Xenova models have onnx/ subdirectory)
        model_dir = Path(model_path)
        onnx_files = list(model_dir.rglob("*.onnx"))  # Recursive glob
        
        if onnx_files:
            logger.info(f"✅ Found {len(onnx_files)} ONNX file(s):")
            for onnx_file in onnx_files[:5]:  # Show first 5
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                rel_path = onnx_file.relative_to(model_dir)
                logger.info(f"   - {rel_path} ({size_mb:.1f} MB)")
            if len(onnx_files) > 5:
                logger.info(f"   ... and {len(onnx_files) - 5} more")
        else:
            logger.warning("⚠️  No ONNX files found in downloaded model")
            logger.warning("   This model may not have ONNX files available")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False


def list_available_models():
    """Print available models."""
    print("\n📦 Available ONNX Models:\n")
    for model_name, info in AVAILABLE_MODELS.items():
        print(f"  {model_name}")
        print(f"    Description: {info['description']}")
        print(f"    Dimensions: {info['dimensions']}")
        print(f"    Size: ~{info['size_mb']} MB")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX embedding models from HuggingFace"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-mpnet-base-v2"),
        help="Model name to download (default: from EMBEDDING_MODEL_NAME env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("EMBEDDING_MODEL_PATH", "models/embedding"),
        help="Output directory (default: from EMBEDDING_MODEL_PATH env var)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return 0
    
    # Convert to Path
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*70)
    print("  ONNX Embedding Model Download")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Output: {output_dir.absolute()}")
    
    if args.model in AVAILABLE_MODELS:
        info = AVAILABLE_MODELS[args.model]
        print(f"\nModel Info:")
        print(f"  - {info['description']}")
        print(f"  - Dimensions: {info['dimensions']}")
        print(f"  - Size: ~{info['size_mb']} MB")
    
    print("\n" + "-"*70 + "\n")
    
    # Download model
    success = download_model(args.model, output_dir)
    
    if success:
        print("\n" + "="*70)
        print("  ✅ Download Complete!")
        print("="*70)
        print(f"\nModel saved to: {output_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Verify .env has correct EMBEDDING_MODEL_NAME and EMBEDDING_MODEL_PATH")
        print("  2. Run tests: pytest tests/unit/services/embedding/test_onnx_local_provider.py")
        print("\n")
        return 0
    else:
        print("\n" + "="*70)
        print("  ❌ Download Failed")
        print("="*70)
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify the model name is correct")
        print("  3. Try: pip install --upgrade huggingface-hub")
        print("  4. Use --list to see available models")
        print("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
