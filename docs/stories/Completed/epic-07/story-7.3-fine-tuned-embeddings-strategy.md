# Story 7.3: Implement Fine-Tuned Embeddings Strategy

**Story ID:** 7.3
**Epic:** Epic 7 - Advanced & Experimental Strategies
**Story Points:** 21
**Priority:** Experimental
**Dependencies:** Epic 3 (Embedding Service), Training infrastructure

---

## User Story

**As a** system
**I want** to use domain-specific embedding models
**So that** accuracy improves for specialized use cases

---

## Overview

Fine-tuned embeddings leverage domain-specific training to improve retrieval accuracy for specialized contexts (medical, legal, scientific, etc.). This strategy provides infrastructure for:

1. Loading and using custom fine-tuned embedding models
2. Training/fine-tuning pipelines
3. A/B testing to compare models
4. Model versioning and rollback
5. Performance metrics tracking

---

## Detailed Requirements

### Functional Requirements

1. **Custom Embedding Model Support**
   - Load custom fine-tuned embedding models
   - Support multiple model formats:
     - Hugging Face models
     - Sentence-Transformers models
     - Custom PyTorch models
     - ONNX models for inference optimization
   - Configure model-specific parameters (pooling, normalization)
   - Handle model initialization and caching
   - Support for quantized models (int8, fp16)

2. **Model Registry**
   - Centralized registry for embedding models
   - Store models with metadata:
     - Model name, version, training date
     - Base model, fine-tuning dataset
     - Performance metrics
     - Model size, inference time
     - Domain/use case tags
   - CRUD operations for models
   - Search and filter models by criteria
   - Model download and caching

3. **Model Versioning**
   - Semantic versioning for models (v1.0.0, v1.1.0, etc.)
   - Track model lineage (base → fine-tuned v1 → v2)
   - Rollback to previous versions
   - Compare versions side-by-side
   - Deprecate old model versions
   - Automatic migration when model updates

4. **A/B Testing Framework**
   - Run multiple embedding models in parallel
   - Split traffic between models (e.g., 80% base, 20% fine-tuned)
   - Track performance metrics per model:
     - Retrieval accuracy (MRR, NDCG, recall@k)
     - Latency (p50, p95, p99)
     - User engagement metrics
   - Statistical significance testing
   - Automated winner selection
   - Gradual traffic shifting (canary deployments)

5. **Fine-Tuning Pipeline**
   - Prepare training data from existing queries/documents
   - Support fine-tuning strategies:
     - **Contrastive Learning**: Positive/negative pairs
     - **Triplet Loss**: Anchor/positive/negative triplets
     - **Hard Negative Mining**: Focus on difficult examples
     - **Knowledge Distillation**: Learn from larger models
   - Hyperparameter optimization
   - Training progress monitoring
   - Validation and early stopping
   - Export trained models

6. **Performance Metrics Tracking**
   - Track embedding quality metrics:
     - Embedding space properties (isotropy, alignment)
     - Clustering quality (silhouette score)
     - Retrieval performance (accuracy, recall)
   - Compare fine-tuned vs base model
   - Track metrics over time
   - Generate comparison reports
   - Visualization of embeddings (t-SNE, UMAP)

7. **Model Evaluation**
   - Offline evaluation on test sets
   - Online evaluation (live traffic)
   - Domain-specific benchmarks
   - Cross-domain evaluation (generalization)
   - Error analysis and failure cases
   - Embedding visualization and inspection

### Non-Functional Requirements

1. **Performance**
   - Model loading: <2s for typical model
   - Inference time comparable to base model (<10% overhead)
   - Support batch inference for efficiency
   - Model caching to avoid reloading

2. **Scalability**
   - Support multiple models concurrently
   - Handle A/B testing traffic split efficiently
   - Scale to production query volumes
   - Distributed training support (optional)

3. **Reliability**
   - Graceful fallback to base model on errors
   - Model health checks
   - Automatic rollback on performance degradation
   - Monitoring and alerting

4. **Flexibility**
   - Pluggable model backends
   - Configurable fine-tuning strategies
   - Customizable evaluation metrics
   - Support for domain-specific preprocessing

5. **Observability**
   - Log model loading and inference
   - Track A/B test results
   - Monitor model performance metrics
   - Export metrics to monitoring systems

---

## Acceptance Criteria

### AC1: Custom Model Support
- [ ] Load custom Hugging Face models
- [ ] Load Sentence-Transformers models
- [ ] Support ONNX models
- [ ] Configure model parameters (pooling, normalization)
- [ ] Model caching working

### AC2: Model Registry
- [ ] Registry stores models with metadata
- [ ] CRUD operations implemented
- [ ] Search and filter functionality
- [ ] Model download and caching
- [ ] At least 3 models registered for testing

### AC3: Model Versioning
- [ ] Semantic versioning implemented
- [ ] Version history tracked
- [ ] Rollback functionality working
- [ ] Version comparison supported

### AC4: A/B Testing Framework
- [ ] Traffic splitting between models
- [ ] Metrics tracked per model
- [ ] Statistical testing implemented
- [ ] Gradual rollout supported
- [ ] Automated winner selection

### AC5: Fine-Tuning Pipeline
- [ ] Training data preparation implemented
- [ ] At least 2 fine-tuning strategies supported
- [ ] Training monitoring working
- [ ] Model export functional

### AC6: Performance Metrics
- [ ] At least 5 metrics tracked
- [ ] Metrics comparison implemented
- [ ] Visualization support
- [ ] Reports generated

### AC7: Model Evaluation
- [ ] Offline evaluation working
- [ ] Online evaluation implemented
- [ ] Benchmark suite created
- [ ] Error analysis tools provided

### AC8: Testing
- [ ] Unit tests for all components (>85% coverage)
- [ ] Integration tests with real models
- [ ] A/B testing simulation
- [ ] Performance benchmarks
- [ ] Model comparison tests

---

## Technical Specifications

### File Structure
```
rag_factory/
├── models/
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base embedding model interface
│   │   ├── custom_model.py          # Custom model wrapper
│   │   ├── registry.py              # Model registry
│   │   ├── versioning.py            # Model versioning
│   │   └── loader.py                # Model loading utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Fine-tuning trainer
│   │   ├── data_prep.py             # Training data preparation
│   │   ├── losses.py                # Loss functions (contrastive, triplet)
│   │   ├── strategies.py            # Training strategies
│   │   └── config.py                # Training configuration
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Model evaluator
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── ab_testing.py            # A/B testing framework
│   │   └── benchmarks.py            # Benchmark suites
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── tracker.py               # Metrics tracker
│       ├── visualizer.py            # Embedding visualization
│       └── reporter.py              # Report generation
│
tests/
├── unit/
│   └── models/
│       ├── embedding/
│       │   ├── test_registry.py
│       │   ├── test_versioning.py
│       │   └── test_loader.py
│       ├── training/
│       │   └── test_trainer.py
│       └── evaluation/
│           ├── test_metrics.py
│           └── test_ab_testing.py
│
├── integration/
│   └── models/
│       └── test_fine_tuned_embeddings_integration.py
```

### Dependencies
```python
# requirements.txt additions
sentence-transformers==2.2.2       # For embedding models
transformers==4.36.0               # Hugging Face models
torch==2.1.2                       # PyTorch
onnx==1.15.0                       # ONNX runtime
onnxruntime==1.16.3                # ONNX inference
mlflow==2.9.2                      # Model tracking & registry
scipy==1.11.4                      # Statistical tests
umap-learn==0.5.5                  # Embedding visualization
```

### Data Models
```python
# rag_factory/models/embedding/models.py
from typing import List, Dict, Any, Optional
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
    """Metadata for an embedding model."""
    model_id: str
    model_name: str
    version: str
    format: ModelFormat
    base_model: Optional[str] = None
    fine_tuned_on: Optional[str] = None  # Dataset name
    domain: Optional[str] = None  # e.g., "medical", "legal"

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
    """Configuration for loading a model."""
    model_path: str
    model_format: ModelFormat
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize_embeddings: bool = True
    use_fp16: bool = False
    use_onnx: bool = False
    device: str = "cpu"
    batch_size: int = 32
    additional_config: Dict[str, Any] = Field(default_factory=dict)

class ABTestConfig(BaseModel):
    """Configuration for A/B testing."""
    test_name: str
    model_a_id: str  # Control model
    model_b_id: str  # Experiment model
    traffic_split: float = 0.5  # Percentage to model B (0.0-1.0)
    duration_days: int = 7
    minimum_samples: int = 1000
    metrics_to_track: List[str] = Field(
        default_factory=lambda: ["latency", "accuracy", "recall@5"]
    )
    statistical_threshold: float = 0.05  # p-value threshold

class ABTestResult(BaseModel):
    """Results from A/B test."""
    test_name: str
    model_a_id: str
    model_b_id: str

    # Sample counts
    model_a_samples: int
    model_b_samples: int

    # Metrics
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {model_a: val, model_b: val}

    # Statistical tests
    p_values: Dict[str, float]  # metric_name -> p_value
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Decision
    winner: Optional[str] = None  # "model_a", "model_b", or "no_difference"
    recommendation: str = ""

    start_time: datetime
    end_time: datetime
```

### Model Registry
```python
# rag_factory/models/embedding/registry.py
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
from .models import EmbeddingModelMetadata, ModelFormat

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for embedding models."""

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "models.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.models = {
                    model_id: EmbeddingModelMetadata(**metadata)
                    for model_id, metadata in data.items()
                }
        else:
            self.models: Dict[str, EmbeddingModelMetadata] = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.metadata_file, "w") as f:
            data = {
                model_id: metadata.dict()
                for model_id, metadata in self.models.items()
            }
            json.dump(data, f, indent=2, default=str)

    def register_model(self, metadata: EmbeddingModelMetadata) -> None:
        """Register a new model."""
        logger.info(f"Registering model: {metadata.model_id}")

        if metadata.model_id in self.models:
            logger.warning(f"Model {metadata.model_id} already exists, overwriting")

        self.models[metadata.model_id] = metadata
        self._save_registry()

    def get_model(self, model_id: str) -> Optional[EmbeddingModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def list_models(
        self,
        domain: Optional[str] = None,
        format: Optional[ModelFormat] = None,
        tags: Optional[List[str]] = None
    ) -> List[EmbeddingModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())

        if domain:
            models = [m for m in models if m.domain == domain]

        if format:
            models = [m for m in models if m.format == format]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            logger.info(f"Deleted model: {model_id}")
            return True
        return False

    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for a model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        for key, value in metrics.items():
            if hasattr(model, key):
                setattr(model, key, value)

        self._save_registry()
        logger.info(f"Updated metrics for model {model_id}")

    def search_models(self, query: str) -> List[EmbeddingModelMetadata]:
        """Search models by name or description."""
        query_lower = query.lower()
        results = []

        for model in self.models.values():
            if (query_lower in model.model_name.lower() or
                (model.description and query_lower in model.description.lower())):
                results.append(model)

        return results
```

Due to length constraints, let me continue this in the next section with the remaining components...


### Custom Model Loader
```python
# rag_factory/models/embedding/loader.py
from typing import Optional
import logging
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from .models import ModelConfig, ModelFormat, PoolingStrategy
import onnxruntime as ort

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """Load custom embedding models."""

    def __init__(self):
        self.loaded_models = {}  # Cache loaded models

    def load_model(self, config: ModelConfig) -> Any:
        """
        Load embedding model based on configuration.

        Args:
            config: Model configuration

        Returns:
            Loaded model object
        """
        logger.info(f"Loading model from: {config.model_path}")

        # Check cache
        cache_key = f"{config.model_path}_{config.model_format.value}"
        if cache_key in self.loaded_models:
            logger.info("Using cached model")
            return self.loaded_models[cache_key]

        # Load based on format
        if config.model_format == ModelFormat.SENTENCE_TRANSFORMERS:
            model = self._load_sentence_transformer(config)
        elif config.model_format == ModelFormat.HUGGINGFACE:
            model = self._load_huggingface(config)
        elif config.model_format == ModelFormat.ONNX:
            model = self._load_onnx(config)
        else:
            raise ValueError(f"Unsupported model format: {config.model_format}")

        # Cache model
        self.loaded_models[cache_key] = model

        logger.info(f"Model loaded successfully: {config.model_format.value}")
        return model

    def _load_sentence_transformer(self, config: ModelConfig) -> SentenceTransformer:
        """Load Sentence-Transformers model."""
        model = SentenceTransformer(config.model_path, device=config.device)

        if config.use_fp16:
            model = model.half()

        return model

    def _load_huggingface(self, config: ModelConfig) -> Dict[str, Any]:
        """Load Hugging Face model."""
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModel.from_pretrained(config.model_path)

        model.to(config.device)
        model.eval()

        if config.use_fp16:
            model = model.half()

        return {"model": model, "tokenizer": tokenizer, "config": config}

    def _load_onnx(self, config: ModelConfig) -> ort.InferenceSession:
        """Load ONNX model."""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if config.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        session = ort.InferenceSession(
            config.model_path,
            sess_options=session_options,
            providers=providers
        )

        return session

    def embed_texts(
        self,
        texts: List[str],
        model: Any,
        config: ModelConfig
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using loaded model.

        Args:
            texts: List of texts to embed
            model: Loaded model
            config: Model configuration

        Returns:
            List of embedding vectors
        """
        if config.model_format == ModelFormat.SENTENCE_TRANSFORMERS:
            embeddings = model.encode(
                texts,
                batch_size=config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=config.normalize_embeddings
            )
            return embeddings.tolist()

        elif config.model_format == ModelFormat.HUGGINGFACE:
            return self._embed_huggingface(texts, model, config)

        elif config.model_format == ModelFormat.ONNX:
            return self._embed_onnx(texts, model, config)

        else:
            raise ValueError(f"Unsupported model format: {config.model_format}")

    def _embed_huggingface(
        self,
        texts: List[str],
        model_dict: Dict[str, Any],
        config: ModelConfig
    ) -> List[List[float]]:
        """Generate embeddings using Hugging Face model."""
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i:i + config.batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(config.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Pool embeddings
            batch_embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                config.pooling_strategy
            )

            # Normalize if configured
            if config.normalize_embeddings:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            embeddings.extend(batch_embeddings.cpu().tolist())

        return embeddings

    def _embed_onnx(
        self,
        texts: List[str],
        session: ort.InferenceSession,
        config: ModelConfig
    ) -> List[List[float]]:
        """Generate embeddings using ONNX model."""
        # ONNX embedding implementation
        # Would need tokenizer and proper input preparation
        raise NotImplementedError("ONNX embedding not yet implemented")

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: PoolingStrategy
    ) -> torch.Tensor:
        """Pool token embeddings into single embedding."""
        if strategy == PoolingStrategy.MEAN:
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif strategy == PoolingStrategy.CLS:
            # Use [CLS] token
            return token_embeddings[:, 0, :]

        elif strategy == PoolingStrategy.MAX:
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
```

### A/B Testing Framework
```python
# rag_factory/models/evaluation/ab_testing.py
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from scipy import stats
from .models import ABTestConfig, ABTestResult
from datetime import datetime

logger = logging.getLogger(__name__)

class ABTestingFramework:
    """Framework for A/B testing embedding models."""

    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}  # test_name -> results

    def start_test(self, config: ABTestConfig) -> None:
        """Start an A/B test."""
        logger.info(f"Starting A/B test: {config.test_name}")

        self.active_tests[config.test_name] = config
        self.results[config.test_name] = []

    def should_use_model_b(self, test_name: str) -> bool:
        """Determine if should use model B for this request (traffic splitting)."""
        if test_name not in self.active_tests:
            return False

        config = self.active_tests[test_name]
        return np.random.random() < config.traffic_split

    def record_result(
        self,
        test_name: str,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Record result from a single request."""
        if test_name not in self.active_tests:
            logger.warning(f"Test {test_name} not active")
            return

        result = {
            "model_id": model_id,
            "timestamp": datetime.now(),
            "metrics": metrics
        }

        self.results[test_name].append(result)

    def analyze_test(self, test_name: str) -> ABTestResult:
        """
        Analyze A/B test results.

        Args:
            test_name: Name of the test

        Returns:
            Test results with statistical analysis
        """
        logger.info(f"Analyzing A/B test: {test_name}")

        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        config = self.active_tests[test_name]
        results = self.results[test_name]

        # Split results by model
        model_a_results = [r for r in results if r["model_id"] == config.model_a_id]
        model_b_results = [r for r in results if r["model_id"] == config.model_b_id]

        # Check minimum samples
        if len(model_a_results) < config.minimum_samples or len(model_b_results) < config.minimum_samples:
            logger.warning(
                f"Insufficient samples: A={len(model_a_results)}, B={len(model_b_results)} "
                f"(minimum={config.minimum_samples})"
            )

        # Calculate metrics for each model
        metrics_comparison = {}
        p_values = {}
        confidence_intervals = {}

        for metric_name in config.metrics_to_track:
            # Extract metric values
            model_a_values = [r["metrics"].get(metric_name) for r in model_a_results if metric_name in r["metrics"]]
            model_b_values = [r["metrics"].get(metric_name) for r in model_b_results if metric_name in r["metrics"]]

            if not model_a_values or not model_b_values:
                continue

            # Calculate means
            model_a_mean = np.mean(model_a_values)
            model_b_mean = np.mean(model_b_values)

            metrics_comparison[metric_name] = {
                "model_a": model_a_mean,
                "model_b": model_b_mean,
                "improvement": ((model_b_mean - model_a_mean) / model_a_mean) * 100
            }

            # Statistical test (t-test)
            t_stat, p_value = stats.ttest_ind(model_a_values, model_b_values)
            p_values[metric_name] = p_value

            # Confidence interval for difference
            diff = np.array(model_b_values) - np.array(model_a_values)
            ci = stats.t.interval(
                0.95,
                len(diff) - 1,
                loc=np.mean(diff),
                scale=stats.sem(diff)
            )
            confidence_intervals[metric_name] = ci

        # Determine winner
        winner, recommendation = self._determine_winner(
            metrics_comparison,
            p_values,
            config.statistical_threshold
        )

        # Create result object
        result = ABTestResult(
            test_name=test_name,
            model_a_id=config.model_a_id,
            model_b_id=config.model_b_id,
            model_a_samples=len(model_a_results),
            model_b_samples=len(model_b_results),
            metrics=metrics_comparison,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            winner=winner,
            recommendation=recommendation,
            start_time=results[0]["timestamp"] if results else datetime.now(),
            end_time=results[-1]["timestamp"] if results else datetime.now()
        )

        logger.info(f"Test analysis complete. Winner: {winner}")

        return result

    def _determine_winner(
        self,
        metrics: Dict[str, Dict[str, float]],
        p_values: Dict[str, float],
        threshold: float
    ) -> Tuple[Optional[str], str]:
        """Determine winner based on statistical significance."""
        significant_improvements = 0
        significant_degradations = 0

        for metric_name, values in metrics.items():
            p_value = p_values.get(metric_name)
            if p_value is None:
                continue

            improvement = values["improvement"]

            # Check statistical significance
            if p_value < threshold:
                if improvement > 0:
                    significant_improvements += 1
                else:
                    significant_degradations += 1

        # Decision logic
        if significant_improvements > significant_degradations:
            return "model_b", "Model B shows statistically significant improvement"
        elif significant_degradations > significant_improvements:
            return "model_a", "Model A performs better, stay with current model"
        else:
            return "no_difference", "No statistically significant difference detected"

    def gradual_rollout(
        self,
        test_name: str,
        new_traffic_split: float
    ) -> None:
        """Gradually increase traffic to model B."""
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        config = self.active_tests[test_name]
        config.traffic_split = max(0.0, min(1.0, new_traffic_split))

        logger.info(
            f"Updated traffic split for {test_name}: "
            f"{config.traffic_split * 100:.1f}% to model B"
        )
```

---

## Unit Tests

### Test File Locations
- `tests/unit/models/embedding/test_registry.py`
- `tests/unit/models/embedding/test_loader.py`
- `tests/unit/models/evaluation/test_ab_testing.py`

### Test Cases

#### TC7.3.1: Model Registry Tests
```python
import pytest
from pathlib import Path
from rag_factory.models.embedding.registry import ModelRegistry
from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelFormat

@pytest.fixture
def temp_registry(tmp_path):
    """Create temporary registry for testing."""
    return ModelRegistry(registry_path=str(tmp_path / "test_registry"))

@pytest.fixture
def sample_metadata():
    """Sample model metadata."""
    return EmbeddingModelMetadata(
        model_id="test_model_v1",
        model_name="Test Embedding Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512,
        domain="general",
        tags=["test", "general"]
    )

def test_register_model(temp_registry, sample_metadata):
    """Test registering a new model."""
    temp_registry.register_model(sample_metadata)

    # Verify model is in registry
    retrieved = temp_registry.get_model("test_model_v1")
    assert retrieved is not None
    assert retrieved.model_name == "Test Embedding Model"

def test_list_models(temp_registry, sample_metadata):
    """Test listing all models."""
    temp_registry.register_model(sample_metadata)

    models = temp_registry.list_models()
    assert len(models) == 1
    assert models[0].model_id == "test_model_v1"

def test_filter_by_domain(temp_registry):
    """Test filtering models by domain."""
    # Register models with different domains
    medical_model = EmbeddingModelMetadata(
        model_id="medical_v1",
        model_name="Medical Model",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
        embedding_dim=768,
        max_seq_length=512,
        domain="medical"
    )

    legal_model = EmbeddingModelMetadata(
        model_id="legal_v1",
        model_name="Legal Model",
        version="1.0.0",
        format=ModelFormat.HUGGINGFACE,
        embedding_dim=768,
        max_seq_length=512,
        domain="legal"
    )

    temp_registry.register_model(medical_model)
    temp_registry.register_model(legal_model)

    # Filter by domain
    medical_models = temp_registry.list_models(domain="medical")
    assert len(medical_models) == 1
    assert medical_models[0].model_id == "medical_v1"

def test_update_metrics(temp_registry, sample_metadata):
    """Test updating model metrics."""
    temp_registry.register_model(sample_metadata)

    # Update metrics
    temp_registry.update_metrics(
        "test_model_v1",
        {
            "retrieval_accuracy": 0.85,
            "mrr_score": 0.78
        }
    )

    # Verify update
    model = temp_registry.get_model("test_model_v1")
    assert model.retrieval_accuracy == 0.85
    assert model.mrr_score == 0.78

def test_search_models(temp_registry, sample_metadata):
    """Test searching models."""
    temp_registry.register_model(sample_metadata)

    results = temp_registry.search_models("test embedding")
    assert len(results) == 1
    assert results[0].model_id == "test_model_v1"

def test_delete_model(temp_registry, sample_metadata):
    """Test deleting a model."""
    temp_registry.register_model(sample_metadata)

    success = temp_registry.delete_model("test_model_v1")
    assert success == True

    # Verify deletion
    model = temp_registry.get_model("test_model_v1")
    assert model is None
```

#### TC7.3.2: A/B Testing Tests
```python
import pytest
from rag_factory.models.evaluation.ab_testing import ABTestingFramework
from rag_factory.models.embedding.models import ABTestConfig

@pytest.fixture
def ab_framework():
    return ABTestingFramework()

@pytest.fixture
def test_config():
    return ABTestConfig(
        test_name="base_vs_finetuned",
        model_a_id="base_model",
        model_b_id="finetuned_model",
        traffic_split=0.5,
        duration_days=7,
        minimum_samples=100,
        metrics_to_track=["latency", "accuracy"],
        statistical_threshold=0.05
    )

def test_start_test(ab_framework, test_config):
    """Test starting an A/B test."""
    ab_framework.start_test(test_config)

    assert "base_vs_finetuned" in ab_framework.active_tests
    assert ab_framework.active_tests["base_vs_finetuned"].model_a_id == "base_model"

def test_traffic_splitting(ab_framework, test_config):
    """Test traffic splitting between models."""
    ab_framework.start_test(test_config)

    # Run many trials
    model_b_count = 0
    trials = 10000

    for _ in range(trials):
        if ab_framework.should_use_model_b("base_vs_finetuned"):
            model_b_count += 1

    # Should be approximately 50% (within reasonable margin)
    model_b_ratio = model_b_count / trials
    assert 0.45 < model_b_ratio < 0.55

def test_record_results(ab_framework, test_config):
    """Test recording test results."""
    ab_framework.start_test(test_config)

    # Record some results
    ab_framework.record_result(
        "base_vs_finetuned",
        "base_model",
        {"latency": 50.0, "accuracy": 0.80}
    )

    ab_framework.record_result(
        "base_vs_finetuned",
        "finetuned_model",
        {"latency": 45.0, "accuracy": 0.85}
    )

    assert len(ab_framework.results["base_vs_finetuned"]) == 2

def test_analyze_test(ab_framework, test_config):
    """Test analyzing A/B test results."""
    ab_framework.start_test(test_config)

    # Record many results with clear difference
    for i in range(200):
        if i < 100:
            # Model A: higher latency, lower accuracy
            ab_framework.record_result(
                "base_vs_finetuned",
                "base_model",
                {"latency": 50.0 + np.random.randn() * 2, "accuracy": 0.80 + np.random.randn() * 0.02}
            )
        else:
            # Model B: lower latency, higher accuracy
            ab_framework.record_result(
                "base_vs_finetuned",
                "finetuned_model",
                {"latency": 45.0 + np.random.randn() * 2, "accuracy": 0.85 + np.random.randn() * 0.02}
            )

    # Analyze
    result = ab_framework.analyze_test("base_vs_finetuned")

    assert result.winner == "model_b"  # Model B should win
    assert result.model_a_samples == 100
    assert result.model_b_samples == 100

def test_gradual_rollout(ab_framework, test_config):
    """Test gradual traffic increase."""
    ab_framework.start_test(test_config)

    # Initial split is 50%
    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 0.5

    # Increase to 80%
    ab_framework.gradual_rollout("base_vs_finetuned", 0.8)

    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 0.8
```

---

## Integration Tests

### Test File Location
`tests/integration/models/test_fine_tuned_embeddings_integration.py`

### Test Scenarios

#### IS7.3.1: End-to-End Model Registration and Usage
```python
import pytest
from rag_factory.models.embedding import ModelRegistry, CustomModelLoader
from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelConfig, ModelFormat

@pytest.mark.integration
def test_register_and_load_model(tmp_path):
    """Test complete workflow of registering and loading a model."""
    # Setup registry
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))

    # Register a model (using a real Sentence-Transformers model)
    metadata = EmbeddingModelMetadata(
        model_id="all-MiniLM-L6-v2",
        model_name="All MiniLM L6 v2",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=256
    )

    registry.register_model(metadata)

    # Load the model
    loader = CustomModelLoader()
    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    model = loader.load_model(config)

    # Generate embeddings
    texts = ["Hello world", "Test embedding"]
    embeddings = loader.embed_texts(texts, model, config)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # Correct dimension

@pytest.mark.integration
def test_ab_testing_workflow(tmp_path):
    """Test A/B testing with two models."""
    from rag_factory.models.evaluation.ab_testing import ABTestingFramework
    from rag_factory.models.embedding.models import ABTestConfig

    framework = ABTestingFramework()

    # Start A/B test
    config = ABTestConfig(
        test_name="model_comparison",
        model_a_id="base_model",
        model_b_id="fine_tuned",
        traffic_split=0.5,
        minimum_samples=50
    )

    framework.start_test(config)

    # Simulate requests with different performance
    for i in range(100):
        if framework.should_use_model_b("model_comparison"):
            # Fine-tuned model: better performance
            framework.record_result(
                "model_comparison",
                "fine_tuned",
                {"latency": 45.0, "accuracy": 0.88}
            )
        else:
            # Base model: baseline performance
            framework.record_result(
                "model_comparison",
                "base_model",
                {"latency": 50.0, "accuracy": 0.82}
            )

    # Analyze results
    result = framework.analyze_test("model_comparison")

    # Fine-tuned model should win
    assert result.winner in ["model_b", "no_difference"]
    assert result.model_a_samples > 0
    assert result.model_b_samples > 0
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_model_comparison_performance.py
import pytest
import time
from rag_factory.models.embedding import CustomModelLoader
from rag_factory.models.embedding.models import ModelConfig, ModelFormat

@pytest.mark.benchmark
def test_model_loading_speed():
    """Benchmark model loading time."""
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    start = time.time()
    model = loader.load_model(config)
    duration = time.time() - start

    print(f"\nModel loading time: {duration:.3f}s")
    assert duration < 2.0, f"Loading too slow: {duration:.2f}s"

@pytest.mark.benchmark
def test_inference_speed_comparison():
    """Compare inference speed of different models."""
    loader = CustomModelLoader()

    # Base model
    base_config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    base_model = loader.load_model(base_config)

    # Test data
    texts = ["Test sentence " * 10] * 100

    # Benchmark base model
    start = time.time()
    base_embeddings = loader.embed_texts(texts, base_model, base_config)
    base_duration = time.time() - start

    print(f"\nBase model: {len(texts)} texts in {base_duration:.3f}s")

    # Calculate throughput
    throughput = len(texts) / base_duration
    print(f"Throughput: {throughput:.1f} texts/second")

    assert throughput > 10, f"Too slow: {throughput:.1f} texts/s"
```

---

## Definition of Done

- [ ] Model registry implemented
- [ ] Custom model loader working
- [ ] Support for 3+ model formats
- [ ] Model versioning implemented
- [ ] A/B testing framework complete
- [ ] Traffic splitting working
- [ ] Statistical analysis implemented
- [ ] Fine-tuning pipeline implemented (basic)
- [ ] Performance metrics tracking
- [ ] Model comparison tools
- [ ] Gradual rollout functionality
- [ ] All unit tests pass (>85% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks
- [ ] Documentation complete
- [ ] Code reviewed

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install sentence-transformers transformers torch onnx onnxruntime mlflow scipy umap-learn

# Optional: Install MLflow for model registry
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Configuration

```yaml
# config.yaml
models:
  embedding:
    # Default model
    default_model_id: "base_model_v1"

    # Model registry
    registry_path: "./model_registry"

    # A/B testing
    ab_testing:
      enabled: true
      default_traffic_split: 0.2  # 20% to new model

    # Model cache
    cache_dir: "./model_cache"
    cache_size_gb: 10
```

### Usage Example

```python
from rag_factory.models.embedding import ModelRegistry, CustomModelLoader, ABTestingFramework
from rag_factory.models.embedding.models import (
    EmbeddingModelMetadata, ModelConfig, ModelFormat, ABTestConfig
)

# 1. Register models
registry = ModelRegistry()

# Register base model
base_metadata = EmbeddingModelMetadata(
    model_id="base_model_v1",
    model_name="Base Embedding Model",
    version="1.0.0",
    format=ModelFormat.SENTENCE_TRANSFORMERS,
    embedding_dim=384,
    max_seq_length=512
)
registry.register_model(base_metadata)

# Register fine-tuned model
finetuned_metadata = EmbeddingModelMetadata(
    model_id="medical_finetuned_v1",
    model_name="Medical Fine-tuned Model",
    version="1.0.0",
    format=ModelFormat.SENTENCE_TRANSFORMERS,
    base_model="base_model_v1",
    fine_tuned_on="medical_qa_dataset",
    embedding_dim=384,
    max_seq_length=512,
    domain="medical"
)
registry.register_model(finetuned_metadata)

# 2. Load models
loader = CustomModelLoader()

base_config = ModelConfig(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    model_format=ModelFormat.SENTENCE_TRANSFORMERS,
    device="cpu"
)
base_model = loader.load_model(base_config)

finetuned_config = ModelConfig(
    model_path="./models/medical_finetuned",
    model_format=ModelFormat.SENTENCE_TRANSFORMERS,
    device="cpu"
)
finetuned_model = loader.load_model(finetuned_config)

# 3. Setup A/B test
ab_framework = ABTestingFramework()

test_config = ABTestConfig(
    test_name="base_vs_medical",
    model_a_id="base_model_v1",
    model_b_id="medical_finetuned_v1",
    traffic_split=0.2,  # 20% to fine-tuned
    minimum_samples=1000,
    metrics_to_track=["latency", "accuracy", "user_satisfaction"]
)

ab_framework.start_test(test_config)

# 4. Use in production
def embed_query(query: str):
    """Embed query using A/B testing."""
    use_finetuned = ab_framework.should_use_model_b("base_vs_medical")

    if use_finetuned:
        embeddings = loader.embed_texts([query], finetuned_model, finetuned_config)
        model_id = "medical_finetuned_v1"
    else:
        embeddings = loader.embed_texts([query], base_model, base_config)
        model_id = "base_model_v1"

    # Record metrics
    ab_framework.record_result(
        "base_vs_medical",
        model_id,
        {"latency": 45.0, "accuracy": 0.85}  # Measure actual metrics
    )

    return embeddings[0]

# 5. Analyze results
result = ab_framework.analyze_test("base_vs_medical")
print(f"Winner: {result.winner}")
print(f"Recommendation: {result.recommendation}")

# 6. Gradual rollout if winner
if result.winner == "model_b":
    ab_framework.gradual_rollout("base_vs_medical", 0.5)  # Increase to 50%
```

---

## Notes for Developers

1. **Model Selection**: Fine-tune only when base models show clear limitations. Measure baseline performance first.

2. **Training Data Quality**: Fine-tuning quality depends entirely on training data. Ensure:
   - High-quality positive/negative pairs
   - Representative domain coverage
   - Sufficient data volume (>10K pairs minimum)

3. **A/B Testing Duration**: Run tests for at least 1 week to account for weekly patterns. Ensure statistical power.

4. **Model Versioning**: Always version models. Never overwrite production models without backup.

5. **Performance Monitoring**: Track both quality (accuracy) and operational (latency, cost) metrics.

6. **Rollback Strategy**: Have automated rollback triggers if new model degrades performance.

7. **Cost Considerations**:
   - Fine-tuning costs (compute, time)
   - Inference costs (may be higher for larger models)
   - Storage costs (multiple model versions)

8. **Domain Adaptation**: Fine-tuned models may perform worse on out-of-domain queries. Monitor generalization.

9. **Maintenance**: Models degrade over time. Plan for periodic retraining.

10. **Evaluation**: Use multiple metrics. A single metric can be misleading.

11. **Privacy**: Be careful with training data. Ensure compliance with data regulations.

12. **Model Zoo**: Maintain a registry of models for different domains. Facilitate reuse.
