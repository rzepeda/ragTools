"""Verification script for fine-tuned embeddings infrastructure.

This script demonstrates the implementation without requiring heavy dependencies.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_registry():
    """Test ModelRegistry functionality."""
    print("=" * 60)
    print("Testing ModelRegistry")
    print("=" * 60)
    
    from rag_factory.models.embedding.registry import ModelRegistry
    from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelFormat
    
    # Create temporary registry
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_path=tmpdir)
        
        # Register a model
        metadata = EmbeddingModelMetadata(
            model_id="test_model_v1",
            model_name="Test Embedding Model",
            version="1.0.0",
            format=ModelFormat.SENTENCE_TRANSFORMERS,
            embedding_dim=384,
            max_seq_length=512,
            domain="general",
            tags=["test", "general"]
        )
        
        registry.register_model(metadata)
        print(f"✓ Registered model: {metadata.model_id}")
        
        # Retrieve model
        retrieved = registry.get_model("test_model_v1")
        assert retrieved is not None
        assert retrieved.model_name == "Test Embedding Model"
        print(f"✓ Retrieved model: {retrieved.model_name}")
        
        # List models
        models = registry.list_models()
        assert len(models) == 1
        print(f"✓ Listed {len(models)} model(s)")
        
        # Update metrics
        registry.update_metrics("test_model_v1", {
            "retrieval_accuracy": 0.85,
            "mrr_score": 0.78
        })
        updated = registry.get_model("test_model_v1")
        assert updated.retrieval_accuracy == 0.85
        print(f"✓ Updated metrics: accuracy={updated.retrieval_accuracy}")
        
        # Search
        results = registry.search_models("test embedding")
        assert len(results) == 1
        print(f"✓ Search found {len(results)} result(s)")
        
        # Delete
        success = registry.delete_model("test_model_v1")
        assert success is True
        assert registry.get_model("test_model_v1") is None
        print(f"✓ Deleted model successfully")
        
    print("\n✅ ModelRegistry tests passed!\n")


def test_ab_testing_framework():
    """Test ABTestingFramework functionality."""
    print("=" * 60)
    print("Testing ABTestingFramework")
    print("=" * 60)
    
    # Check if scipy is available
    try:
        from scipy import stats
        scipy_available = True
    except ImportError:
        scipy_available = False
        print("⚠️  scipy not available, skipping statistical tests")
        return
    
    from rag_factory.models.evaluation.ab_testing import ABTestingFramework
    from rag_factory.models.evaluation.models import ABTestConfig
    import numpy as np
    
    framework = ABTestingFramework()
    
    # Start test
    config = ABTestConfig(
        test_name="base_vs_finetuned",
        model_a_id="base_model",
        model_b_id="finetuned_model",
        traffic_split=0.5,
        minimum_samples=50
    )
    
    framework.start_test(config)
    print(f"✓ Started A/B test: {config.test_name}")
    
    # Test traffic splitting
    model_b_count = sum(
        1 for _ in range(1000)
        if framework.should_use_model_b("base_vs_finetuned")
    )
    ratio = model_b_count / 1000
    assert 0.4 < ratio < 0.6
    print(f"✓ Traffic split working: {ratio:.1%} to model B")
    
    # Record results
    np.random.seed(42)
    for i in range(100):
        if i < 50:
            framework.record_result(
                "base_vs_finetuned",
                "base_model",
                {"latency": 50.0 + np.random.randn() * 2, "accuracy": 0.80}
            )
        else:
            framework.record_result(
                "base_vs_finetuned",
                "finetuned_model",
                {"latency": 45.0 + np.random.randn() * 2, "accuracy": 0.85}
            )
    
    print(f"✓ Recorded 100 results")
    
    # Analyze
    result = framework.analyze_test("base_vs_finetuned")
    print(f"✓ Analysis complete:")
    print(f"  - Model A samples: {result.model_a_samples}")
    print(f"  - Model B samples: {result.model_b_samples}")
    print(f"  - Winner: {result.winner}")
    print(f"  - Recommendation: {result.recommendation}")
    
    # Gradual rollout
    framework.gradual_rollout("base_vs_finetuned", 0.8)
    assert framework.active_tests["base_vs_finetuned"].traffic_split == 0.8
    print(f"✓ Gradual rollout to 80%")
    
    print("\n✅ ABTestingFramework tests passed!\n")


def test_data_models():
    """Test data models."""
    print("=" * 60)
    print("Testing Data Models")
    print("=" * 60)
    
    from rag_factory.models.embedding.models import (
        ModelFormat,
        PoolingStrategy,
        EmbeddingModelMetadata,
        ModelConfig
    )
    from rag_factory.models.evaluation.models import ABTestConfig, ABTestResult
    from datetime import datetime
    
    # Test ModelFormat enum
    assert ModelFormat.SENTENCE_TRANSFORMERS == "sentence_transformers"
    assert ModelFormat.HUGGINGFACE == "huggingface"
    print("✓ ModelFormat enum working")
    
    # Test PoolingStrategy enum
    assert PoolingStrategy.MEAN == "mean"
    assert PoolingStrategy.CLS == "cls"
    print("✓ PoolingStrategy enum working")
    
    # Test EmbeddingModelMetadata
    metadata = EmbeddingModelMetadata(
        model_id="test",
        model_name="Test Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512
    )
    assert metadata.model_id == "test"
    print("✓ EmbeddingModelMetadata working")
    
    # Test ModelConfig
    config = ModelConfig(
        model_path="path/to/model",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS
    )
    assert config.normalize_embeddings is True  # default
    assert config.batch_size == 32  # default
    print("✓ ModelConfig working")
    
    # Test ABTestConfig
    ab_config = ABTestConfig(
        test_name="test",
        model_a_id="a",
        model_b_id="b"
    )
    assert ab_config.traffic_split == 0.5  # default
    print("✓ ABTestConfig working")
    
    # Test ABTestResult
    ab_result = ABTestResult(
        test_name="test",
        model_a_id="a",
        model_b_id="b",
        model_a_samples=100,
        model_b_samples=100,
        metrics={},
        p_values={},
        confidence_intervals={},
        start_time=datetime.now(),
        end_time=datetime.now()
    )
    assert ab_result.test_name == "test"
    print("✓ ABTestResult working")
    
    print("\n✅ Data models tests passed!\n")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("Fine-Tuned Embeddings Infrastructure Verification")
    print("=" * 60 + "\n")
    
    try:
        test_data_models()
        test_model_registry()
        test_ab_testing_framework()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNote: Full integration tests with real models require:")
        print("  - torch")
        print("  - sentence-transformers")
        print("  - scipy")
        print("  - onnx/onnxruntime (for ONNX support)")
        print("\nInstall with: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
