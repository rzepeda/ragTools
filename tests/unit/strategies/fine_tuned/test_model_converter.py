import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

try:
    import torch
    from scripts.convert_finetuned_to_onnx import convert_finetuned_to_onnx, validate_onnx_model
except ImportError:
    torch = None

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
class TestModelConverter:
    """Test model conversion script."""

    @patch("scripts.convert_finetuned_to_onnx.torch.load")
    @patch("scripts.convert_finetuned_to_onnx.torch.onnx.export")
    def test_convert_finetuned_to_onnx(self, mock_export, mock_load, tmp_path):
        """Test conversion function."""
        model_path = tmp_path / "model.pt"
        output_path = tmp_path / "model.onnx"
        
        # Mock model
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        convert_finetuned_to_onnx(model_path, output_path)
        
        mock_load.assert_called_with(model_path, map_location="cpu")
        mock_model.eval.assert_called_once()
        mock_export.assert_called_once()

    @patch("scripts.convert_finetuned_to_onnx.torch.load")
    @patch("scripts.convert_finetuned_to_onnx.InferenceSession")
    def test_validate_onnx_model(self, mock_session_cls, mock_load, tmp_path):
        """Test validation function."""
        model_path = tmp_path / "model.pt"
        onnx_path = tmp_path / "model.onnx"
        
        # Mock PyTorch model
        mock_pytorch_model = MagicMock()
        mock_pytorch_output = MagicMock()
        mock_pytorch_output.last_hidden_state.numpy.return_value = torch.ones((1, 10)).numpy()
        mock_pytorch_model.return_value = mock_pytorch_output
        mock_load.return_value = mock_pytorch_model
        
        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.run.return_value = [torch.ones((1, 10)).numpy()] # Exact match
        mock_session_cls.return_value = mock_session
        
        result = validate_onnx_model(onnx_path, model_path)
        
        assert result is True
        
        # Test failure case
        mock_session.run.return_value = [torch.zeros((1, 10)).numpy()] # Mismatch
        
        result = validate_onnx_model(onnx_path, model_path)
        
        assert result is False
