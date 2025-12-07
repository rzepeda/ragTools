"""Integration tests for fine-tuned embeddings infrastructure."""

import pytest
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
    """Test complete workflow of registering and loading a model."""
    # Setup registry
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))

    # Register a model (using a real Sentence-Transformers model)
    metadata = EmbeddingModelMetadata(
        model_id="all-MiniLM-L6-v2",
        model_name="All MiniLM L6 v2",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=256
    )

    registry.register_model(metadata)

    # Verify registration
    retrieved = registry.get_model("all-MiniLM-L6-v2")
    assert retrieved is not None
    assert retrieved.embedding_dim == 384

    # Load the model
    loader = CustomModelLoader()
    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

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
        model_a_id="base_model",
        model_b_id="fine_tuned",
        traffic_split=0.5,
        minimum_samples=50
    )

    framework.start_test(config)

    # Simulate requests with different performance
    import numpy as np
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
    
    metadata1 = EmbeddingModelMetadata(
        model_id="model1",
        model_name="Model 1",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512,
        domain="general"
    )
    
    metadata2 = EmbeddingModelMetadata(
        model_id="model2",
        model_name="Model 2",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
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
    """Test comparing two models end-to-end."""
    # Setup
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))
    loader = CustomModelLoader()
    ab_framework = ABTestingFramework()
    
    # Register base model
    base_metadata = EmbeddingModelMetadata(
        model_id="base_model",
        model_name="Base Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=256
    )
    registry.register_model(base_metadata)
    
    # Load model
    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )
    model = loader.load_model(config)
    
    # Start A/B test (comparing same model against itself for testing)
    test_config = ABTestConfig(
        test_name="comparison",
        model_a_id="base_model",
        model_b_id="base_model",  # Same model for testing
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
    
    # Register multiple models
    models = [
        EmbeddingModelMetadata(
            model_id=f"model_{i}",
            model_name=f"Model {i}",
            version="1.0.0",
            format=ModelFormat.SENTENCE_TRANSFORMERS,
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
