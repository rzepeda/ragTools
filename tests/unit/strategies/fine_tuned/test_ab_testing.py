import pytest
from unittest.mock import MagicMock
from datetime import datetime
from rag_factory.strategies.fine_tuned.ab_testing import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentResult
)

class TestABTestingFramework:
    """Test A/B Testing Framework."""

    @pytest.fixture
    def mock_loader(self):
        """Mock model loader."""
        loader = MagicMock()
        loader.load_model.return_value = "mock_model"
        return loader

    @pytest.fixture
    def framework(self, mock_loader):
        """Create framework."""
        return ABTestingFramework(mock_loader)

    def test_start_stop_experiment(self, framework):
        """Test starting and stopping experiments."""
        config = ExperimentConfig(
            experiment_id="exp1",
            model_a_id="model-a",
            model_b_id="model-b",
            start_time=datetime.now()
        )
        
        framework.start_experiment(config)
        assert "exp1" in framework.active_experiments
        
        framework.stop_experiment("exp1")
        assert "exp1" not in framework.active_experiments

    def test_get_model_for_request(self, framework):
        """Test model selection."""
        config = ExperimentConfig(
            experiment_id="exp1",
            model_a_id="model-a",
            model_b_id="model-b",
            traffic_split=0.5,
            start_time=datetime.now()
        )
        framework.start_experiment(config)
        
        # Test multiple calls to verify split (probabilistic)
        # We mock random to ensure deterministic testing
        with pytest.MonkeyPatch.context() as m:
            # Force Model A (random > split)
            m.setattr("random.random", lambda: 0.6)
            model, model_id, version = framework.get_model_for_request("exp1")
            assert model_id == "model-a"
            
            # Force Model B (random < split)
            m.setattr("random.random", lambda: 0.4)
            model, model_id, version = framework.get_model_for_request("exp1")
            assert model_id == "model-b"

    def test_record_and_stats(self, framework):
        """Test recording results and getting stats."""
        config = ExperimentConfig(
            experiment_id="exp1",
            model_a_id="model-a",
            model_b_id="model-b",
            start_time=datetime.now()
        )
        framework.start_experiment(config)
        
        # Record results for A
        framework.record_result(ExperimentResult(
            experiment_id="exp1",
            model_id="model-a",
            version="1.0",
            timestamp=datetime.now(),
            duration_ms=10.0,
            output="out"
        ))
        
        # Record results for B
        framework.record_result(ExperimentResult(
            experiment_id="exp1",
            model_id="model-b",
            version="1.0",
            timestamp=datetime.now(),
            duration_ms=20.0,
            output="out"
        ))
        
        stats = framework.get_experiment_stats("exp1")
        
        assert stats["total_requests"] == 2
        assert stats["model_a_requests"] == 1
        assert stats["model_b_requests"] == 1
        assert stats["avg_latency_a"] == 10.0
        assert stats["avg_latency_b"] == 20.0
