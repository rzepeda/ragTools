"""Data models for embedding model infrastructure."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ModelFormat(str, Enum):
    """Supported model formats."""
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class PoolingStrategy(str, Enum):
    """Token pooling strategies."""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    WEIGHTED_MEAN = "weighted_mean"


class EmbeddingModelMetadata(BaseModel):
    """Metadata for an embedding model.
    
    Attributes:
        model_id: Unique identifier for the model
        model_name: Human-readable name
        version: Semantic version (e.g., "1.0.0")
        format: Model format (HuggingFace, Sentence-Transformers, etc.)
        base_model: Base model name if fine-tuned
        fine_tuned_on: Dataset name used for fine-tuning
        domain: Domain/use case (e.g., "medical", "legal")
        embedding_dim: Embedding vector dimension
        max_seq_length: Maximum sequence length
        avg_inference_time_ms: Average inference time in milliseconds
        model_size_mb: Model size in megabytes
        retrieval_accuracy: Retrieval accuracy metric
        mrr_score: Mean Reciprocal Rank score
        ndcg_at_10: NDCG@10 score
        created_at: Creation timestamp
        created_by: Creator identifier
        description: Model description
        tags: List of tags for categorization
    """
    model_id: str
    model_name: str
    version: str
    format: ModelFormat
    base_model: Optional[str] = None
    fine_tuned_on: Optional[str] = None
    domain: Optional[str] = None

    # Performance metrics
    embedding_dim: int
    max_seq_length: int
    avg_inference_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None

    # Quality metrics
    retrieval_accuracy: Optional[float] = None
    mrr_score: Optional[float] = None
    ndcg_at_10: Optional[float] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Configuration for loading a model.
    
    Attributes:
        model_path: Path to model (local path or HuggingFace model ID)
        model_format: Format of the model
        pooling_strategy: Strategy for pooling token embeddings
        normalize_embeddings: Whether to normalize embeddings
        use_fp16: Use FP16 precision for inference
        use_onnx: Use ONNX runtime for inference
        device: Device to run on ("cpu", "cuda", "mps")
        batch_size: Batch size for inference
        additional_config: Additional configuration parameters
    """
    model_path: str
    model_format: ModelFormat
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize_embeddings: bool = True
    use_fp16: bool = False
    use_onnx: bool = False
    device: str = "cpu"
    batch_size: int = 32
    additional_config: Dict[str, Any] = Field(default_factory=dict)
