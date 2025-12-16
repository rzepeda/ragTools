from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional

class FineTunedConfig(BaseSettings):
    """Configuration for fine-tuned embeddings strategy."""
    
    registry_dir: Optional[Path] = Field(
        default=None,
        description="Directory for model registry"
    )
    
    default_model_id: str = Field(
        default="default-model",
        description="Default model ID to load"
    )
    
    prefer_onnx: bool = Field(
        default=True,
        description="Whether to prefer ONNX models over PyTorch"
    )
    
    enable_ab_testing: bool = Field(
        default=False,
        description="Enable A/B testing"
    )

    class Config:
        env_prefix = "RAG_FINE_TUNED_"
