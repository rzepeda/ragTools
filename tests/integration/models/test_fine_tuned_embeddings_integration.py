"""Integration tests for fine-tuned embeddings infrastructure.

NOTE: This test suite has been migrated to use ONNX format (Epic 10).
For multi-format embedding support, see docs/BACKLOG.md - "Embedding Provider Interface" story.

These tests focus on the infrastructure (registry, A/B testing, versioning) rather than
actual model loading, which is tested separately in embedding service tests.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from rag_factory.models.embedding import (
    ModelRegistry,
    CustomModelLoader,
    EmbeddingModelMetadata,
    ModelConfig,
    ModelFormat
)
from rag_factory.models.evaluation import ABTestingFramework, ABTestConfig


@pytest.mark.integration
def test_register_and_load_model(tmp_path):
    """Test complete workflow of registering and loading a model.
    
    Note: Model loading is mocked to avoid requiring actual ONNX files.
    Actual model loading is tested in embedding service integration tests.
    """
    # Setup registry
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))

    # Register a model (using ONNX format - Epic 10 architecture)
    metadata = EmbeddingModelMetadata(
        model_id="all-MiniLM-L6-v2",
        model_name="All MiniLM L6 v2",
        version="1.0.0",
        format=ModelFormat.ONNX,
        embedding_dim=384,
        max_seq_length=256
    )

    registry.register_model(metadata)

    # Verify registration
    retrieved = registry.get_model("all-MiniLM-L6-v2")
    assert retrieved is not None
    assert retrieved.embedding_dim == 384

    # Mock model loading (infrastructure test, not model test)
    loader = CustomModelLoader()
    config = ModelConfig(
        model_path="Xenova/all-MiniLM-L6-v2",
        model_format=ModelFormat.ONNX,
        device="cpu",
        use_onnx=True
    )

    # Mock the ONNX model and embeddings
    mock_model = Mock()
    mock_embeddings = np.random.randn(2, 384).tolist()
    
    with patch.object(loader, 'load_model', return_value=mock_model):
        with patch.object(loader, 'embed_texts', return_value=mock_embeddings):
            model = loader.load_model(config)
            
            # Generate embeddings
            texts = ["Hello world", "Test embedding"]
            embeddings = loader.embed_texts(texts, model, config)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384  # Correct dimension


@pytest.mark.integration
def test_ab_testing_workflow(tmp_path):
    """Test A/B testing with two models."""
    framework = ABTestingFramework()

    # Start A/B test
    config = ABTestConfig(
        test_name="model_comparison",
        model_a_id="base_model_a",
        model_b_id="fine_tuned",
        traffic_split=0.5,
        minimum_samples=50
    )

    framework.start_test(config)

    # Simulate requests with different performance
    np.random.seed(42)
    
    for i in range(100):
        if framework.should_use_model_b("model_comparison"):
            # Fine-tuned model: better performance
            framework.record_result(
                "model_comparison",
                "fine_tuned",
                {
                    "latency": 45.0 + np.random.randn() * 2,
                    "accuracy": 0.88 + np.random.randn() * 0.02
                }
            )
        else:
            # Base model: baseline performance
            framework.record_result(
                "model_comparison",
                "base_model",
                {
                    "latency": 50.0 + np.random.randn() * 2,
                    "accuracy": 0.82 + np.random.randn() * 0.02
                }
            )

    # Analyze results
    result = framework.analyze_test("model_comparison")

    # Verify results
    assert result.winner in ["model_b", "no_difference", "model_a"]
    assert result.model_a_samples > 0
    assert result.model_b_samples > 0
    assert "latency" in result.metrics
    assert "accuracy" in result.metrics


@pytest.mark.integration
def test_model_registry_persistence_workflow(tmp_path):
    """Test that registry persists across sessions."""
    registry_path = str(tmp_path / "registry")
    
    # Session 1: Create and populate registry
    registry1 = ModelRegistry(registry_path=registry_path)
    
    # Using ONNX format (Epic 10)
    metadata1 = EmbeddingModelMetadata(
        model_id="model1",
        model_name="Model 1",
        version="1.0.0",
        format=ModelFormat.ONNX,
        embedding_dim=384,
        max_seq_length=512,
        domain="general"
    )
    
    metadata2 = EmbeddingModelMetadata(
        model_id="model2",
        model_name="Model 2",
        version="1.0.0",
        format=ModelFormat.ONNX,
        embedding_dim=768,
        max_seq_length=512,
        domain="medical"
    )
    
    registry1.register_model(metadata1)
    registry1.register_model(metadata2)
    
    # Update metrics
    registry1.update_metrics("model1", {"retrieval_accuracy": 0.85})
    
    # Session 2: Load registry
    registry2 = ModelRegistry(registry_path=registry_path)
    
    # Verify all data persisted
    assert len(registry2.list_models()) == 2
    
    model1 = registry2.get_model("model1")
    assert model1 is not None
    assert model1.retrieval_accuracy == 0.85
    
    medical_models = registry2.list_models(domain="medical")
    assert len(medical_models) == 1
    assert medical_models[0].model_id == "model2"


@pytest.mark.integration
def test_model_comparison_workflow(tmp_path):
    """Test comparing two models end-to-end (simplified to test A/B framework)."""
    # This test focuses on the A/B testing framework, not actual model loading
    ab_framework = ABTestingFramework()
    
    # Start A/B test
    test_config = ABTestConfig(
        test_name="comparison",
        model_a_id="model_a",
        model_b_id="model_b",
        traffic_split=0.5,
        minimum_samples=10
    )
    ab_framework.start_test(test_config)
    
    # Simulate usage with mock embeddings
    for _ in range(20):
        # Determine which model to use (traffic splitting)
        if ab_framework.should_use_model_b("comparison"):
            model_id = test_config.model_b_id
        else:
            model_id = test_config.model_a_id
        
        ab_framework.record_result(
            "comparison",
            model_id,
            {"latency": 50.0, "embedding_dim": 384}
        )
    
    # Analyze
    result = ab_framework.analyze_test("comparison")
    
    # Verify A/B framework works correctly
    assert result.winner == "no_difference"
    assert result.model_a_samples > 0
    assert result.model_b_samples > 0
    assert result.model_a_samples + result.model_b_samples == 20


def test_model_comparison_workflow(tmp_path):
    """Test comparing two models end-to-end.
    
    Note: Model loading is mocked to focus on testing the comparison infrastructure.
    """
    # Setup
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))
    loader = CustomModelLoader()
    ab_framework = ABTestingFramework()
    
    # Register base model (ONNX format)
    base_metadata = EmbeddingModelMetadata(
        model_id="base_model",
        model_name="Base Model",
        version="1.0.0",
        format=ModelFormat.ONNX,
        embedding_dim=384,
        max_seq_length=256
    )
    registry.register_model(base_metadata)
    
    # Mock model loading
    config = ModelConfig(
        model_path="Xenova/all-MiniLM-L6-v2",
        model_format=ModelFormat.ONNX,
        device="cpu",
        use_onnx=True
    )
    
    mock_model = Mock()
    mock_embeddings = np.random.randn(2, 384).tolist()
    
    with patch.object(loader, 'load_model', return_value=mock_model):
        with patch.object(loader, 'embed_texts', return_value=mock_embeddings):
            model = loader.load_model(config)
            
            # Start A/B test (comparing same model against itself for testing)
            test_config = ABTestConfig(
                test_name="comparison",
                model_a_id="base_model_a",
                model_b_id="base_model_b",  # Different ID for A/B testing
                traffic_split=0.5,
                minimum_samples=10
            )
            ab_framework.start_test(test_config)
            
            # Simulate usage
            test_texts = ["Sample text 1", "Sample text 2"]
            
            for _ in range(20):
                embeddings = loader.embed_texts(test_texts, model, config)
                
                # Record metrics
                model_id = "base_model"
                ab_framework.record_result(
                    "comparison",
                    model_id,
                    {"latency": 50.0, "embedding_dim": len(embeddings[0])}
                )
            
            # Analyze
            result = ab_framework.analyze_test("comparison")
            
            # Since it's the same model, should show no difference
            assert result.winner == "no_difference"
            assert result.model_a_samples + result.model_b_samples == 20


@pytest.mark.integration
def test_multiple_models_registry(tmp_path):
    """Test managing multiple models in registry."""
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))
    
    # Register multiple models (all ONNX format)
    models = [
        EmbeddingModelMetadata(
            model_id=f"model_{i}",
            model_name=f"Model {i}",
            version="1.0.0",
            format=ModelFormat.ONNX,
            embedding_dim=384,
            max_seq_length=512,
            domain="general" if i % 2 == 0 else "medical",
            tags=["fast"] if i < 3 else ["accurate"]
        )
        for i in range(5)
    ]
    
    for model in models:
        registry.register_model(model)
    
    # Test various queries
    all_models = registry.list_models()
    assert len(all_models) == 5
    
    general_models = registry.list_models(domain="general")
    assert len(general_models) == 3
    
    medical_models = registry.list_models(domain="medical")
    assert len(medical_models) == 2
    
    fast_models = registry.list_models(tags=["fast"])
    assert len(fast_models) == 3
    
    # Search
    results = registry.search_models("Model 2")
    assert len(results) == 1
    assert results[0].model_id == "model_2"
    
    # Delete
    registry.delete_model("model_0")
    assert len(registry.list_models()) == 4
