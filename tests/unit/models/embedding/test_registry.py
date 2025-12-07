"""Unit tests for ModelRegistry."""

import pytest
from pathlib import Path
from rag_factory.models.embedding.registry import ModelRegistry
from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelFormat


@pytest.fixture
def temp_registry(tmp_path):
    """Create temporary registry for testing."""
    return ModelRegistry(registry_path=str(tmp_path / "test_registry"))


@pytest.fixture
def sample_metadata():
    """Sample model metadata."""
    return EmbeddingModelMetadata(
        model_id="test_model_v1",
        model_name="Test Embedding Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512,
        domain="general",
        tags=["test", "general"]
    )


def test_register_model(temp_registry, sample_metadata):
    """Test registering a new model."""
    temp_registry.register_model(sample_metadata)

    # Verify model is in registry
    retrieved = temp_registry.get_model("test_model_v1")
    assert retrieved is not None
    assert retrieved.model_name == "Test Embedding Model"
    assert retrieved.embedding_dim == 384


def test_register_model_overwrites_existing(temp_registry, sample_metadata):
    """Test that registering same model ID overwrites."""
    temp_registry.register_model(sample_metadata)
    
    # Register again with different name
    sample_metadata.model_name = "Updated Model"
    temp_registry.register_model(sample_metadata)
    
    retrieved = temp_registry.get_model("test_model_v1")
    assert retrieved.model_name == "Updated Model"


def test_get_model_not_found(temp_registry):
    """Test getting non-existent model returns None."""
    result = temp_registry.get_model("nonexistent")
    assert result is None


def test_list_models(temp_registry, sample_metadata):
    """Test listing all models."""
    temp_registry.register_model(sample_metadata)

    models = temp_registry.list_models()
    assert len(models) == 1
    assert models[0].model_id == "test_model_v1"


def test_filter_by_domain(temp_registry):
    """Test filtering models by domain."""
    # Register models with different domains
    medical_model = EmbeddingModelMetadata(
        model_id="medical_v1",
        model_name="Medical Model",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
        embedding_dim=768,
        max_seq_length=512,
        domain="medical"
    )

    legal_model = EmbeddingModelMetadata(
        model_id="legal_v1",
        model_name="Legal Model",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
        embedding_dim=768,
        max_seq_length=512,
        domain="legal"
    )

    temp_registry.register_model(medical_model)
    temp_registry.register_model(legal_model)

    # Filter by domain
    medical_models = temp_registry.list_models(domain="medical")
    assert len(medical_models) == 1
    assert medical_models[0].model_id == "medical_v1"


def test_filter_by_format(temp_registry):
    """Test filtering models by format."""
    st_model = EmbeddingModelMetadata(
        model_id="st_model",
        model_name="ST Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512
    )

    hf_model = EmbeddingModelMetadata(
        model_id="hf_model",
        model_name="HF Model",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
        embedding_dim=768,
        max_seq_length=512
    )

    temp_registry.register_model(st_model)
    temp_registry.register_model(hf_model)

    # Filter by format
    st_models = temp_registry.list_models(format=ModelFormat.SENTENCE_TRANSFORMERS)
    assert len(st_models) == 1
    assert st_models[0].model_id == "st_model"


def test_filter_by_tags(temp_registry):
    """Test filtering models by tags."""
    model1 = EmbeddingModelMetadata(
        model_id="model1",
        model_name="Model 1",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512,
        tags=["fast", "small"]
    )

    model2 = EmbeddingModelMetadata(
        model_id="model2",
        model_name="Model 2",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=768,
        max_seq_length=512,
        tags=["accurate", "large"]
    )

    temp_registry.register_model(model1)
    temp_registry.register_model(model2)

    # Filter by tags
    fast_models = temp_registry.list_models(tags=["fast"])
    assert len(fast_models) == 1
    assert fast_models[0].model_id == "model1"


def test_update_metrics(temp_registry, sample_metadata):
    """Test updating model metrics."""
    temp_registry.register_model(sample_metadata)

    # Update metrics
    temp_registry.update_metrics(
        "test_model_v1",
        {
            "retrieval_accuracy": 0.85,
            "mrr_score": 0.78,
            "ndcg_at_10": 0.82
        }
    )

    # Verify update
    model = temp_registry.get_model("test_model_v1")
    assert model.retrieval_accuracy == 0.85
    assert model.mrr_score == 0.78
    assert model.ndcg_at_10 == 0.82


def test_update_metrics_nonexistent_model(temp_registry):
    """Test updating metrics for non-existent model raises error."""
    with pytest.raises(ValueError, match="Model .* not found"):
        temp_registry.update_metrics("nonexistent", {"retrieval_accuracy": 0.9})


def test_search_models(temp_registry, sample_metadata):
    """Test searching models by name."""
    temp_registry.register_model(sample_metadata)

    results = temp_registry.search_models("test embedding")
    assert len(results) == 1
    assert results[0].model_id == "test_model_v1"


def test_search_models_by_description(temp_registry):
    """Test searching models by description."""
    model = EmbeddingModelMetadata(
        model_id="model1",
        model_name="Model 1",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512,
        description="A model for medical text analysis"
    )

    temp_registry.register_model(model)

    results = temp_registry.search_models("medical")
    assert len(results) == 1
    assert results[0].model_id == "model1"


def test_search_models_case_insensitive(temp_registry, sample_metadata):
    """Test search is case-insensitive."""
    temp_registry.register_model(sample_metadata)

    results = temp_registry.search_models("TEST EMBEDDING")
    assert len(results) == 1


def test_delete_model(temp_registry, sample_metadata):
    """Test deleting a model."""
    temp_registry.register_model(sample_metadata)

    success = temp_registry.delete_model("test_model_v1")
    assert success is True

    # Verify deletion
    model = temp_registry.get_model("test_model_v1")
    assert model is None


def test_delete_nonexistent_model(temp_registry):
    """Test deleting non-existent model returns False."""
    success = temp_registry.delete_model("nonexistent")
    assert success is False


def test_registry_persistence(tmp_path):
    """Test that registry persists to disk."""
    registry_path = str(tmp_path / "test_registry")
    
    # Create registry and add model
    registry1 = ModelRegistry(registry_path=registry_path)
    metadata = EmbeddingModelMetadata(
        model_id="test_model",
        model_name="Test Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512
    )
    registry1.register_model(metadata)

    # Create new registry instance with same path
    registry2 = ModelRegistry(registry_path=registry_path)
    
    # Verify model was loaded
    retrieved = registry2.get_model("test_model")
    assert retrieved is not None
    assert retrieved.model_name == "Test Model"


def test_empty_registry(temp_registry):
    """Test operations on empty registry."""
    models = temp_registry.list_models()
    assert len(models) == 0

    results = temp_registry.search_models("anything")
    assert len(results) == 0
