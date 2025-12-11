#!/usr/bin/env python3
"""Test ONNX embedding model setup.

This script tests that the ONNX embedding model is properly configured
and can generate embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("="*70)
print("  ONNX Embedding Model Test")
print("="*70)

# Check environment variables
print("\n📋 Environment Configuration:")
print(f"  EMBEDDING_MODEL_NAME: {os.getenv('EMBEDDING_MODEL_NAME', 'Not set')}")
print(f"  EMBEDDING_MODEL_PATH: {os.getenv('EMBEDDING_MODEL_PATH', 'Not set')}")

# Try to import and initialize ONNX provider
print("\n🔧 Testing ONNX Provider...")

try:
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    
    # Create provider with minimal config
    provider = ONNXLocalProvider(config={})
    
    print(f"  ✅ Provider initialized successfully")
    print(f"  Model: {provider.get_model_name()}")
    print(f"  Dimensions: {provider.get_dimensions()}")
    print(f"  Max batch size: {provider.get_max_batch_size()}")
    
    # Test embedding generation
    print("\n🧪 Testing Embedding Generation...")
    test_texts = [
        "Hello, world!",
        "This is a test of the ONNX embedding model."
    ]
    
    result = provider.get_embeddings(test_texts)
    
    print(f"  ✅ Generated embeddings for {len(test_texts)} texts")
    print(f"  Embedding dimensions: {len(result.embeddings[0])}")
    print(f"  Token count: {result.token_count}")
    print(f"  Model used: {result.model}")
    
    # Show first few values of first embedding
    print(f"\n  First embedding (first 5 values): {result.embeddings[0][:5]}")
    
    print("\n" + "="*70)
    print("  ✅ All Tests Passed!")
    print("="*70)
    print("\nThe ONNX embedding model is working correctly.")
    print("You can now run the full test suite:")
    print("  pytest tests/unit/services/embedding/test_onnx_local_provider.py -v")
    print()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure the model is downloaded:")
    print("     python scripts/download_embedding_model.py")
    print("  2. Check your .env file has correct variables")
    print("  3. Verify ONNX Runtime is installed:")
    print("     pip install onnxruntime")
    print()
    sys.exit(1)
