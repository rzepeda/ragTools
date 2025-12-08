import pytest
import onnx
from onnx import helper, TensorProto
import numpy as np
from pathlib import Path
from rag_factory.strategies.fine_tuned.model_registry import ONNXModelRegistry
from rag_factory.strategies.fine_tuned.custom_loader import CustomModelLoader
from rag_factory.strategies.fine_tuned.config import FineTunedConfig

def create_dummy_onnx_model(output_path: Path):
    """Create a dummy ONNX model that returns identity."""
    # Define input and output
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    # Create a node (Identity)
    node_def = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input_info],
        [output_info],
    )

    # Create the model
    opset = helper.make_opsetid("", 14)
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-example',
        opset_imports=[opset],
        ir_version=8
    )
    
    # Save
    onnx.save(model_def, str(output_path))

class TestFineTunedONNXIntegration:
    """Integration tests for Fine-Tuned ONNX embeddings."""

    @pytest.fixture
    def registry_dir(self, tmp_path):
        return tmp_path / "registry"

    @pytest.fixture
    def dummy_model_path(self, tmp_path):
        model_path = tmp_path / "dummy.onnx"
        create_dummy_onnx_model(model_path)
        return model_path

    def test_end_to_end_flow(self, registry_dir, dummy_model_path):
        """Test registering, loading, and using an ONNX model."""
        
        # 1. Initialize Registry
        registry = ONNXModelRegistry(registry_dir=registry_dir)
        
        # 2. Register Model
        metadata = registry.register_model(
            model_id="dummy-model",
            model_path=dummy_model_path,
            version="1.0.0",
            format="onnx",
            description="Dummy integration model",
            validate=True
        )
        
        assert metadata.model_id == "dummy-model"
        
        # 3. Initialize Loader
        config = FineTunedConfig(
            registry_dir=registry_dir,
            default_model_id="dummy-model",
            prefer_onnx=True
        )
        loader = CustomModelLoader(config)
        
        # 4. Load Model
        session = loader.load_model("dummy-model")
        
        # 5. Run Inference
        input_data = np.random.randn(1, 10).astype(np.float32)
        output = session.run(None, {'input': input_data})[0]
        
        # Verify output (Identity model)
        np.testing.assert_allclose(output, input_data)
        
        # 6. Verify Metadata
        _, loaded_meta = registry.get_model("dummy-model")
        assert loaded_meta.version == "1.0.0"
