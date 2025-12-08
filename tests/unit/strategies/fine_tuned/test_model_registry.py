import pytest
from pathlib import Path
from datetime import datetime
from rag_factory.strategies.fine_tuned.model_registry import (
    ONNXModelRegistry,
    ModelMetadata
)

class TestONNXModelRegistry:
    """Test ONNX model registry."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create temporary registry."""
        return ONNXModelRegistry(registry_dir=tmp_path)

    @pytest.fixture
    def mock_onnx_model(self, tmp_path):
        """Create mock ONNX model file."""
        model_path = tmp_path / "test_model.onnx"
        model_path.write_text("mock onnx model")
        return model_path

    def test_register_model(self, registry, mock_onnx_model):
        """Test model registration."""
        # Mock validation to avoid needing real ONNX model
        with pytest.MonkeyPatch.context() as m:
            m.setattr(registry, "_validate_model", lambda *args: None)
            m.setattr(registry, "_get_embedding_dim", lambda *args: 384)
            
            metadata = registry.register_model(
                model_id="test-model",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="onnx",
                description="Test model",
                validate=True
            )

        assert metadata.model_id == "test-model"
        assert metadata.version == "1.0.0"
        assert metadata.format == "onnx"
        assert metadata.embedding_dim == 384
        
        # Verify file copied
        saved_path = registry.registry_dir / "test-model" / "1.0.0" / "model.onnx"
        assert saved_path.exists()
        assert saved_path.read_text() == "mock onnx model"

    def test_get_model(self, registry, mock_onnx_model):
        """Test model retrieval."""
        # Register first
        with pytest.MonkeyPatch.context() as m:
            m.setattr(registry, "_validate_model", lambda *args: None)
            m.setattr(registry, "_get_embedding_dim", lambda *args: 384)
            
            registry.register_model(
                model_id="test-model",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="onnx",
                validate=True
            )

        # Get specific version
        model_path, metadata = registry.get_model("test-model", version="1.0.0")
        assert model_path.exists()
        assert metadata.version == "1.0.0"

        # Get latest version
        model_path, metadata = registry.get_model("test-model")
        assert metadata.version == "1.0.0"

    def test_list_models(self, registry, mock_onnx_model):
        """Test listing models."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(registry, "_validate_model", lambda *args: None)
            m.setattr(registry, "_get_embedding_dim", lambda *args: 384)
            
            registry.register_model(
                model_id="model1",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="onnx",
                tags=["production"],
                validate=True
            )
            
            registry.register_model(
                model_id="model2",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="pytorch",
                tags=["staging"],
                validate=True
            )

        # List all
        models = registry.list_models()
        assert len(models) == 2

        # Filter by format
        onnx_models = registry.list_models(format_filter="onnx")
        assert len(onnx_models) == 1
        assert onnx_models[0].model_id == "model1"

        # Filter by tag
        prod_models = registry.list_models(tag_filter=["production"])
        assert len(prod_models) == 1
        assert prod_models[0].model_id == "model1"

    def test_delete_model(self, registry, mock_onnx_model):
        """Test model deletion."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(registry, "_validate_model", lambda *args: None)
            m.setattr(registry, "_get_embedding_dim", lambda *args: 384)
            
            registry.register_model(
                model_id="test-model",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="onnx",
                validate=True
            )

        registry.delete_model("test-model", "1.0.0")

        with pytest.raises(ValueError):
            registry.get_model("test-model", version="1.0.0")
            
        # Verify files deleted
        model_dir = registry.registry_dir / "test-model" / "1.0.0"
        assert not model_dir.exists()

    def test_get_latest_version(self, registry, mock_onnx_model):
        """Test getting latest version."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(registry, "_validate_model", lambda *args: None)
            m.setattr(registry, "_get_embedding_dim", lambda *args: 384)
            
            # Register v1.0.0
            registry.register_model(
                model_id="test-model",
                model_path=mock_onnx_model,
                version="1.0.0",
                format="onnx",
                validate=True
            )
            
            # Register v2.0.0
            registry.register_model(
                model_id="test-model",
                model_path=mock_onnx_model,
                version="2.0.0",
                format="onnx",
                validate=True
            )

        _, metadata = registry.get_model("test-model")
        assert metadata.version == "2.0.0"
