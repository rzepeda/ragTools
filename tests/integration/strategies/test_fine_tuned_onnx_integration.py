import pytest
import torch
import sys
from pathlib import Path
import shutil
from rag_factory.strategies.fine_tuned.model_registry import ONNXModelRegistry
from rag_factory.strategies.fine_tuned.custom_loader import CustomModelLoader
from rag_factory.strategies.fine_tuned.config import FineTunedConfig
from rag_factory.strategies.fine_tuned.ab_testing import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentResult
)

# Add scripts to path to import conversion function
sys.path.append(str(Path(__file__).parents[4] / "scripts"))
from convert_finetuned_to_onnx import convert_finetuned_to_onnx

class TestFineTunedONNXIntegration:
    """Integration test for fine-tuned ONNX embeddings."""
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace."""
        return tmp_path
        
    @pytest.fixture
    def dummy_pytorch_model(self, workspace):
        """Create a dummy PyTorch model."""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(30522, 384)
                
            def forward(self, input_ids, attention_mask):
                return self.embeddings(input_ids)
                
        model = DummyModel()
        model_path = workspace / "dummy_model.pt"
        torch.save(model, model_path)
        return model_path
        
    def test_full_flow(self, workspace, dummy_pytorch_model):
        """Test the full flow from conversion to A/B testing."""
        # 1. Convert to ONNX
        onnx_path = workspace / "dummy_model.onnx"
        convert_finetuned_to_onnx(
            model_path=dummy_pytorch_model,
            output_path=onnx_path,
            input_shape=(1, 10) # Small shape for test
        )
        assert onnx_path.exists()
        
        # 2. Register model
        registry_dir = workspace / "registry"
        registry = ONNXModelRegistry(registry_dir=registry_dir)
        
        metadata = registry.register_model(
            model_id="integration-test-model",
            model_path=onnx_path,
            version="1.0.0",
            format="onnx",
            description="Integration test model"
        )
        assert metadata.model_id == "integration-test-model"
        
        # 3. Load model
        config = FineTunedConfig(
            registry_dir=registry_dir,
            default_model_id="integration-test-model",
            prefer_onnx=True
        )
        loader = CustomModelLoader(config)
        
        model = loader.load_model("integration-test-model", "1.0.0")
        assert model is not None
        
        # 4. Run A/B Experiment
        framework = ABTestingFramework(loader)
        
        # Register a second version (same model for simplicity)
        registry.register_model(
            model_id="integration-test-model",
            model_path=onnx_path,
            version="1.0.1",
            format="onnx"
        )
        
        exp_config = ExperimentConfig(
            experiment_id="exp-integration",
            model_a_id="integration-test-model",
            model_a_version="1.0.0",
            model_b_id="integration-test-model",
            model_b_version="1.0.1",
            start_time=pytest.helpers.mock_datetime_now() if hasattr(pytest.helpers, 'mock_datetime_now') else None, # Just pass None or real time
            traffic_split=0.5
        )
        # Fix start_time since mock helper might not exist
        from datetime import datetime
        exp_config.start_time = datetime.now()
        
        framework.start_experiment(exp_config)
        
        # Simulate requests
        for _ in range(10):
            model_instance, mid, ver = framework.get_model_for_request("exp-integration")
            assert mid == "integration-test-model"
            assert ver in ["1.0.0", "1.0.1"]
            
            # Record result
            framework.record_result(ExperimentResult(
                experiment_id="exp-integration",
                model_id=mid,
                version=ver,
                timestamp=datetime.now(),
                duration_ms=5.0,
                output="embedding"
            ))
            
        stats = framework.get_experiment_stats("exp-integration")
        assert stats["total_requests"] == 10
