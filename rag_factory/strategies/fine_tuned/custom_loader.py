import logging
from pathlib import Path
from typing import Optional, Any, Union
import onnxruntime as ort
from .model_registry import ONNXModelRegistry
from .config import FineTunedConfig

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """
    Loader for fine-tuned embedding models.
    Prioritizes ONNX models from the registry.
    """
    
    def __init__(self, config: FineTunedConfig):
        self.config = config
        self.registry = ONNXModelRegistry(registry_dir=self.config.registry_dir)
        self._loaded_models = {}
        
    def load_model(
        self, 
        model_id: Optional[str] = None, 
        version: Optional[str] = None
    ) -> Any:
        """
        Load a model.
        
        Args:
            model_id: Model ID (uses default if None)
            version: Model version (uses latest if None)
            
        Returns:
            Loaded model (ONNX session or PyTorch model)
        """
        model_id = model_id or self.config.default_model_id
        
        # Check cache
        cache_key = f"{model_id}:{version}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
            
        # Get model from registry
        prefer_format = "onnx" if self.config.prefer_onnx else "pytorch"
        try:
            model_path, metadata = self.registry.get_model(
                model_id=model_id,
                version=version,
                prefer_format=prefer_format
            )
        except ValueError:
            # Fallback to other format if preferred not found
            fallback_format = "pytorch" if prefer_format == "onnx" else "onnx"
            logger.info(f"Preferred format {prefer_format} not found, trying {fallback_format}")
            model_path, metadata = self.registry.get_model(
                model_id=model_id,
                version=version,
                prefer_format=fallback_format
            )
            
        logger.info(f"Loading model {model_id} (v{metadata.version}) from {model_path}")
        
        # Load model based on format
        if metadata.format == "onnx":
            model = self._load_onnx(model_path)
        elif metadata.format == "pytorch":
            model = self._load_pytorch(model_path)
        else:
            raise ValueError(f"Unsupported format: {metadata.format}")
            
        # Cache model
        self._loaded_models[cache_key] = model
        return model
        
    def _load_onnx(self, model_path: Path) -> ort.InferenceSession:
        """Load ONNX model."""
        return ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        
    def _load_pytorch(self, model_path: Path) -> Any:
        """Load PyTorch model."""
        try:
            import torch
            return torch.load(model_path, map_location="cpu")
        except ImportError:
            raise ImportError("PyTorch is required to load PyTorch models")
