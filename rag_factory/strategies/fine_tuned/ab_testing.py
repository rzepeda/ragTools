import logging
import random
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
from .custom_loader import CustomModelLoader

logger = logging.getLogger(__name__)

class ExperimentConfig(BaseModel):
    """Configuration for an A/B experiment."""
    experiment_id: str
    model_a_id: str
    model_b_id: str
    model_a_version: Optional[str] = None
    model_b_version: Optional[str] = None
    traffic_split: float = 0.5  # 0.0 to 1.0, fraction of traffic to Model B
    start_time: datetime
    end_time: Optional[datetime] = None
    description: Optional[str] = None
    metrics: List[str] = ["latency", "accuracy"]

class ExperimentResult(BaseModel):
    """Result of an inference in an experiment."""
    experiment_id: str
    model_id: str
    version: str
    timestamp: datetime
    duration_ms: float
    output: Any  # The embedding or result
    error: Optional[str] = None

class ABTestingFramework:
    """
    Framework for A/B testing embedding models.
    Supports comparing ONNX vs PyTorch models.
    """
    
    def __init__(self, loader: CustomModelLoader):
        self.loader = loader
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.results: List[ExperimentResult] = []
        
    def start_experiment(self, config: ExperimentConfig):
        """Start a new experiment."""
        self.active_experiments[config.experiment_id] = config
        logger.info(f"Started experiment: {config.experiment_id}")
        
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
            logger.info(f"Stopped experiment: {experiment_id}")
            
    def get_model_for_request(self, experiment_id: str) -> tuple[Any, str, str]:
        """
        Select model for a request based on traffic split.
        
        Returns:
            Tuple of (model_instance, model_id, version)
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
            
        config = self.active_experiments[experiment_id]
        
        # Determine which model to use
        if random.random() < config.traffic_split:
            # Use Model B
            model_id = config.model_b_id
            version = config.model_b_version
        else:
            # Use Model A
            model_id = config.model_a_id
            version = config.model_a_version
            
        # Load model
        model = self.loader.load_model(model_id, version)
        
        # Get actual version from loader cache or registry if possible
        # For now, we return the requested version or "latest"
        actual_version = version or "latest"
        
        return model, model_id, actual_version
        
    def record_result(self, result: ExperimentResult):
        """Record experiment result."""
        self.results.append(result)
        # In a real system, this would write to a database or metrics system
        
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        exp_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        stats = {
            "total_requests": len(exp_results),
            "model_a_requests": 0,
            "model_b_requests": 0,
            "avg_latency_a": 0.0,
            "avg_latency_b": 0.0,
            "errors_a": 0,
            "errors_b": 0
        }
        
        if not exp_results:
            return stats
            
        config = self.active_experiments.get(experiment_id)
        if not config:
            return stats
            
        latencies_a = []
        latencies_b = []
        
        for r in exp_results:
            if r.model_id == config.model_a_id:
                stats["model_a_requests"] += 1
                if r.error:
                    stats["errors_a"] += 1
                else:
                    latencies_a.append(r.duration_ms)
            elif r.model_id == config.model_b_id:
                stats["model_b_requests"] += 1
                if r.error:
                    stats["errors_b"] += 1
                else:
                    latencies_b.append(r.duration_ms)
                    
        if latencies_a:
            stats["avg_latency_a"] = sum(latencies_a) / len(latencies_a)
        if latencies_b:
            stats["avg_latency_b"] = sum(latencies_b) / len(latencies_b)
            
        return stats
