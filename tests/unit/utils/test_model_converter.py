import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag_factory.services.utils.model_converter import convert_raw_pytorch_to_onnx, validate_conversion


@pytest.fixture
def mock_torch():
    with patch("rag_factory.services.utils.model_converter.torch") as mock:
        yield mock


@pytest.fixture
def mock_onnx_export(mock_torch):
    return mock_torch.onnx.export


def test_convert_raw_pytorch_to_onnx(tmp_path):
    """Test raw PyTorch conversion calls torch.onnx.export."""
    # Create mocks
    mock_torch_module = MagicMock()
    mock_model = MagicMock()
    mock_model.config.type_vocab_size = 2
    mock_torch_module.load.return_value = mock_model
    
    with patch.dict(sys.modules, {'torch': mock_torch_module}):
        model_path = str(tmp_path / "model.pt")
        output_path = str(tmp_path / "model.onnx")
        
        # Create dummy model file
        with open(model_path, "w") as f:
            f.write("dummy")
            
        # Run conversion
        convert_raw_pytorch_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_shape=(1, 128),
            opset_version=13
        )
        
        # Verify export called
        assert mock_torch_module.onnx.export.called
        args, kwargs = mock_torch_module.onnx.export.call_args
        
        assert args[0] == mock_model
        assert kwargs["opset_version"] == 13
        assert "input_ids" in kwargs["input_names"]
        assert "attention_mask" in kwargs["input_names"]


def test_convert_raw_pytorch_to_onnx_no_torch():
    """Test conversion raises ImportError if torch missing."""
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="torch is required"):
            convert_raw_pytorch_to_onnx("model.pt", "model.onnx")


def test_validate_conversion(tmp_path):
    """Test validation logic."""
    # Create mocks
    mock_st_module = MagicMock()
    mock_st_class = MagicMock()
    mock_st_module.SentenceTransformer = mock_st_class
    
    mock_transformers_module = MagicMock()
    mock_tokenizer_class = MagicMock()
    mock_transformers_module.AutoTokenizer = mock_tokenizer_class
    
    mock_optimum_module = MagicMock()
    mock_ort_class = MagicMock()
    mock_optimum_module.ORTModelForFeatureExtraction = mock_ort_class
    
    # Setup return values
    mock_st_instance = mock_st_class.return_value
    mock_st_instance.encode.return_value = [[1.0, 0.0]]  # Original embedding
    
    mock_ort_instance = mock_ort_class.from_pretrained.return_value
    mock_ort_instance.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = [0.9, 0.1]  # ONNX embedding
    
    # Patch modules
    with patch.dict(sys.modules, {
        'sentence_transformers': mock_st_module,
        'transformers': mock_transformers_module,
        'optimum.onnxruntime': mock_optimum_module,
        'torch': MagicMock()
    }):
        # Run validation
        result = validate_conversion(
            original_model_name="test-model",
            onnx_model_path=tmp_path / "onnx_model",
            test_texts=["test"],
            similarity_threshold=0.8
        )
        
        assert result["passed"] is True or result["passed"] == np.True_
        assert result["mean_similarity"] > 0.8
