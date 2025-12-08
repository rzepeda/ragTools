"""Integration tests for fine-tuned ONNX embeddings."""

import pytest
import numpy as np
from pathlib import Path
import shutil
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag_factory.models.embedding.registry import ModelRegistry
from rag_factory.models.embedding.loader import CustomModelLoader
from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelFormat, ModelConfig
from rag_factory.services.utils.model_converter import convert_raw_pytorch_to_onnx

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


if TORCH_AVAILABLE:
    class SimpleModel(nn.Module):
        """Simple BERT-like model for testing."""
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'type_vocab_size': 2})()
            self.embeddings = nn.Embedding(1000, 32)
            self.encoder = nn.Linear(32, 32)
            
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            x = self.embeddings(input_ids)
            x = self.encoder(x)
            return type('Output', (), {'last_hidden_state': x, 'pooler_output': x[:, 0]})()
else:
    SimpleModel = None


@pytest.mark.skipif(not TORCH_AVAILABLE or not ONNX_AVAILABLE, reason="Requires torch and onnxruntime")
def test_full_onnx_workflow(tmp_path):
    """Test full workflow: Convert -> Register -> Load -> Embed."""
    
    # 1. Create dummy PyTorch model
    model_path = tmp_path / "model.pt"
    model = SimpleModel()
    torch.save(model, model_path)
    
    # 2. Convert to ONNX
    onnx_path = tmp_path / "model.onnx"
    convert_raw_pytorch_to_onnx(
        model_path=str(model_path),
        output_path=str(onnx_path),
        input_shape=(1, 10),
        opset_version=14
    )
    
    assert onnx_path.exists()
    
    # 3. Register in Registry
    registry_path = tmp_path / "registry"
    registry = ModelRegistry(registry_path=str(registry_path))
    
    metadata = EmbeddingModelMetadata(
        model_id="test_onnx_model",
        model_name="Test ONNX Model",
        version="1.0.0",
        format=ModelFormat.ONNX,
        embedding_dim=32,
        max_seq_length=512,
        model_path=str(onnx_path)  # In real usage, this might be relative or managed
    )
    registry.register_model(metadata)
    
    retrieved = registry.get_model("test_onnx_model")
    assert retrieved is not None
    
    # 4. Load using CustomModelLoader
    loader = CustomModelLoader()
    config = ModelConfig(
        model_path=str(onnx_path),
        model_format=ModelFormat.ONNX,
        device="cpu",
        tokenizer_name="cl100k_base"  # Use tiktoken for test
    )
    
    loaded_model = loader.load_model(config)
    assert loaded_model is not None
    
    # 5. Generate Embeddings
    texts = ["Hello world", "Testing ONNX"]
    embeddings = loader.embed_texts(texts, loaded_model, config)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 32
    assert isinstance(embeddings[0][0], float)
    
    # Verify values are not all zero
    assert np.any(np.array(embeddings) != 0)
