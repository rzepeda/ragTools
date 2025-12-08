import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rag_factory.strategies.fine_tuned.custom_loader import CustomModelLoader
from rag_factory.strategies.fine_tuned.config import FineTunedConfig
from rag_factory.strategies.fine_tuned.model_registry import ModelMetadata

class TestCustomModelLoader:
    """Test CustomModelLoader."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create config."""
        return FineTunedConfig(
            registry_dir=tmp_path,
            default_model_id="default",
            prefer_onnx=True
        )
        
    @pytest.fixture
    def loader(self, config):
        """Create loader."""
        with patch("rag_factory.strategies.fine_tuned.custom_loader.ONNXModelRegistry") as mock_registry_cls:
            loader = CustomModelLoader(config)
            loader.registry = MagicMock()
            return loader
            
    @patch("rag_factory.strategies.fine_tuned.custom_loader.ort.InferenceSession")
    def test_load_model_onnx(self, mock_session, loader):
        """Test loading ONNX model."""
        # Setup registry mock
        mock_path = Path("/tmp/model.onnx")
        mock_meta = ModelMetadata(
            model_id="test-model",
            version="1.0.0",
            format="onnx",
            embedding_dim=384,
            created_at="2023-01-01T00:00:00"
        )
        loader.registry.get_model.return_value = (mock_path, mock_meta)
        
        # Load
        model = loader.load_model("test-model")
        
        # Verify
        loader.registry.get_model.assert_called_with(
            model_id="test-model",
            version=None,
            prefer_format="onnx"
        )
        mock_session.assert_called_once()
        assert model == mock_session.return_value
        
    def test_load_model_pytorch(self, loader):
        """Test loading PyTorch model."""
        # Setup registry mock
        mock_path = Path("/tmp/model.pt")
        mock_meta = ModelMetadata(
            model_id="test-model",
            version="1.0.0",
            format="pytorch",
            embedding_dim=384,
            created_at="2023-01-01T00:00:00"
        )
        loader.registry.get_model.return_value = (mock_path, mock_meta)
        
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Load
            model = loader.load_model("test-model")
            
            # Verify
            mock_torch.load.assert_called_with(mock_path, map_location="cpu")
            assert model == mock_torch.load.return_value
            
    def test_load_model_fallback(self, loader):
        """Test fallback to other format."""
        # Setup registry mock to fail first, succeed second
        loader.registry.get_model.side_effect = [
            ValueError("Not found"),
            (Path("/tmp/model.pt"), ModelMetadata(
                model_id="test-model",
                version="1.0.0",
                format="pytorch",
                embedding_dim=384,
                created_at="2023-01-01T00:00:00"
            ))
        ]
        
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            loader.load_model("test-model")
            
            # Verify called twice
            assert loader.registry.get_model.call_count == 2
