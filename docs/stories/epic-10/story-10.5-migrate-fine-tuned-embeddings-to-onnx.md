# Story 10.5: Migrate Fine-Tuned Embeddings to ONNX

**Story ID:** 10.5
**Epic:** Epic 10 - Lightweight Dependencies Implementation
**Story Points:** 5
**Priority:** Medium
**Dependencies:** Story 10.1 (ONNX embeddings)

---

## User Story

**As a** developer
**I want** custom embedding models in ONNX format
**So that** fine-tuned models don't require PyTorch

---

## Detailed Requirements

### Functional Requirements

1. **ONNX Model Registry**
   - Update `CustomModelLoader` to prioritize ONNX format
   - Support ONNX model registration and storage
   - Implement model metadata management (version, metrics, etc.)
   - Add model validation on registration
   - Support multiple model formats (ONNX primary, PyTorch optional)
   - Implement model versioning and rollback

2. **Model Loading and Caching**
   - Load ONNX models from registry
   - Implement efficient model caching
   - Support lazy loading for memory efficiency
   - Handle model updates and hot-swapping
   - Validate model compatibility
   - Provide model health checks

3. **A/B Testing Framework Updates**
   - Update A/B testing to work with ONNX models
   - Support comparing ONNX vs PyTorch models
   - Maintain traffic splitting functionality
   - Preserve statistical analysis capabilities
   - Support gradual rollout of ONNX models
   - Track model performance metrics

4. **Model Conversion Tools**
   - Provide PyTorch → ONNX conversion utilities
   - Support conversion of fine-tuned models
   - Validate converted models for accuracy
   - Preserve model weights and architecture
   - Document conversion process
   - Handle edge cases and special layers

5. **Model Versioning**
   - Implement version tracking for ONNX models
   - Support semantic versioning (major.minor.patch)
   - Track model lineage and provenance
   - Enable model comparison across versions
   - Support rollback to previous versions
   - Document version changes

6. **Training and Conversion Workflow**
   - Document fine-tuning workflow
   - Provide conversion best practices
   - Create example conversion scripts
   - Add quality validation steps
   - Document deployment process
   - Provide troubleshooting guide

### Non-Functional Requirements

1. **Performance**
   - Model loading: <5 seconds
   - Inference speed: comparable to PyTorch
   - Memory usage: <1GB per model
   - A/B testing overhead: <10ms

2. **Quality**
   - Converted models: >99% accuracy vs PyTorch
   - No loss of fine-tuning benefits
   - Consistent embedding quality
   - Validated against test sets

3. **Reliability**
   - Robust model validation
   - Graceful handling of corrupted models
   - Automatic fallback to previous version
   - Comprehensive error handling

4. **Maintainability**
   - Clear model registry structure
   - Well-documented conversion process
   - Version control integration
   - Audit trail for model changes

5. **Resource Efficiency**
   - No PyTorch dependency for inference
   - Efficient model storage
   - Optimized model loading
   - Minimal memory overhead

---

## Acceptance Criteria

### AC1: ONNX Model Registry
- [ ] `CustomModelLoader` prioritizes ONNX
- [ ] Model registration working
- [ ] Metadata management implemented
- [ ] Model validation on registration
- [ ] Multiple formats supported
- [ ] Versioning and rollback working

### AC2: Model Loading
- [ ] ONNX models load from registry
- [ ] Caching implemented
- [ ] Lazy loading working
- [ ] Hot-swapping supported
- [ ] Compatibility validation working
- [ ] Health checks implemented

### AC3: A/B Testing Updates
- [ ] A/B testing works with ONNX
- [ ] ONNX vs PyTorch comparison supported
- [ ] Traffic splitting working
- [ ] Statistical analysis preserved
- [ ] Gradual rollout working
- [ ] Metrics tracking implemented

### AC4: Conversion Tools
- [ ] Conversion utility created
- [ ] Fine-tuned models convertible
- [ ] Accuracy validation working
- [ ] Weights preserved correctly
- [ ] Documentation complete
- [ ] Edge cases handled

### AC5: Model Versioning
- [ ] Version tracking implemented
- [ ] Semantic versioning supported
- [ ] Lineage tracking working
- [ ] Version comparison available
- [ ] Rollback working
- [ ] Changes documented

### AC6: Documentation
- [ ] Fine-tuning workflow documented
- [ ] Conversion guide published
- [ ] Example scripts provided
- [ ] Quality validation documented
- [ ] Deployment process documented
- [ ] Troubleshooting guide created

### AC7: Testing
- [ ] Unit tests for model registry
- [ ] Unit tests for conversion
- [ ] Integration tests with A/B testing
- [ ] Quality validation tests
- [ ] All tests passing without PyTorch

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── embeddings/
│   │   ├── fine_tuned/
│   │   │   ├── __init__.py
│   │   │   ├── model_registry.py      # UPDATED: ONNX support
│   │   │   ├── custom_loader.py       # UPDATED: Prioritize ONNX
│   │   │   ├── ab_testing.py          # UPDATED: ONNX support
│   │   │   └── config.py
│   │   │
│   │   └── utils/
│   │       └── model_converter.py     # Conversion utilities

scripts/
├── convert_finetuned_to_onnx.py       # Conversion script
└── validate_onnx_model.py             # Validation script

tests/
├── unit/
│   └── services/
│       └── embeddings/
│           └── fine_tuned/
│               ├── test_model_registry.py
│               ├── test_custom_loader.py
│               ├── test_ab_testing.py
│               └── test_model_converter.py
│
└── integration/
    └── services/
        └── test_fine_tuned_onnx_integration.py
```

### Updated Model Registry
```python
# rag_factory/services/embeddings/fine_tuned/model_registry.py
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from pydantic import BaseModel
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for registered model."""
    model_id: str
    version: str
    format: str  # "onnx" or "pytorch"
    embedding_dim: int
    created_at: datetime
    description: Optional[str] = None
    metrics: Dict[str, float] = {}
    tags: List[str] = []
    parent_version: Optional[str] = None


class ONNXModelRegistry:
    """
    Registry for ONNX embedding models.
    Prioritizes ONNX format, supports PyTorch as fallback.
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory for storing models
        """
        if registry_dir is None:
            registry_dir = Path.home() / ".cache" / "rag_factory" / "model_registry"

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelMetadata] = {}

        self._load_registry()

        logger.info(f"Initialized model registry at: {self.registry_dir}")

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        version: str,
        format: str = "onnx",
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        validate: bool = True
    ) -> ModelMetadata:
        """
        Register a model in the registry.

        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            version: Model version (semantic versioning)
            format: Model format ("onnx" or "pytorch")
            description: Model description
            metrics: Performance metrics
            tags: Model tags
            validate: Whether to validate model

        Returns:
            Model metadata
        """
        # Validate model if requested
        if validate:
            self._validate_model(model_path, format)

        # Get embedding dimension
        embedding_dim = self._get_embedding_dim(model_path, format)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            format=format,
            embedding_dim=embedding_dim,
            created_at=datetime.now(),
            description=description,
            metrics=metrics or {},
            tags=tags or []
        )

        # Copy model to registry
        model_dir = self.registry_dir / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest_path = model_dir / f"model.{format}"
        shutil.copy(model_path, dest_path)

        # Store metadata
        key = f"{model_id}:{version}"
        self.models[key] = metadata
        self._save_registry()

        logger.info(f"Registered model: {key} ({format})")

        return metadata

    def get_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        prefer_format: str = "onnx"
    ) -> tuple[Path, ModelMetadata]:
        """
        Get model from registry.

        Args:
            model_id: Model identifier
            version: Specific version (None = latest)
            prefer_format: Preferred format

        Returns:
            Tuple of (model_path, metadata)
        """
        # Get version
        if version is None:
            version = self._get_latest_version(model_id, prefer_format)

        key = f"{model_id}:{version}"

        if key not in self.models:
            raise ValueError(f"Model not found: {key}")

        metadata = self.models[key]

        # Get model path
        model_dir = self.registry_dir / model_id / version
        model_path = model_dir / f"model.{metadata.format}"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return model_path, metadata

    def list_models(
        self,
        format_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List registered models.

        Args:
            format_filter: Filter by format
            tag_filter: Filter by tags

        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        # Apply filters
        if format_filter:
            models = [m for m in models if m.format == format_filter]

        if tag_filter:
            models = [
                m for m in models
                if any(tag in m.tags for tag in tag_filter)
            ]

        return models

    def delete_model(self, model_id: str, version: str):
        """Delete model from registry."""
        key = f"{model_id}:{version}"

        if key not in self.models:
            raise ValueError(f"Model not found: {key}")

        # Delete files
        model_dir = self.registry_dir / model_id / version
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)

        # Remove from registry
        del self.models[key]
        self._save_registry()

        logger.info(f"Deleted model: {key}")

    def _validate_model(self, model_path: Path, format: str):
        """Validate model file."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if format == "onnx":
            # Try to load with ONNX Runtime
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"ONNX model validated: {model_path}")
            except Exception as e:
                raise ValueError(f"Invalid ONNX model: {e}")

        elif format == "pytorch":
            # Try to load with PyTorch (if available)
            try:
                import torch
                model = torch.load(model_path, map_location="cpu")
                logger.info(f"PyTorch model validated: {model_path}")
            except ImportError:
                logger.warning("PyTorch not available, skipping validation")
            except Exception as e:
                raise ValueError(f"Invalid PyTorch model: {e}")

    def _get_embedding_dim(self, model_path: Path, format: str) -> int:
        """Get embedding dimension from model."""
        if format == "onnx":
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"]
            )
            output_meta = session.get_outputs()[0]
            return output_meta.shape[-1]

        elif format == "pytorch":
            # Would need to inspect PyTorch model
            # For now, return default
            return 384

        return 384

    def _get_latest_version(
        self,
        model_id: str,
        prefer_format: str = "onnx"
    ) -> str:
        """Get latest version of a model."""
        # Filter models by ID
        model_versions = [
            (key.split(":")[1], meta)
            for key, meta in self.models.items()
            if key.startswith(f"{model_id}:")
        ]

        if not model_versions:
            raise ValueError(f"No versions found for model: {model_id}")

        # Prefer specified format
        preferred = [
            (ver, meta) for ver, meta in model_versions
            if meta.format == prefer_format
        ]

        if preferred:
            model_versions = preferred

        # Sort by version (simple string sort for now)
        model_versions.sort(key=lambda x: x[0], reverse=True)

        return model_versions[0][0]

    def _load_registry(self):
        """Load registry from disk."""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)

            for key, meta_dict in data.items():
                self.models[key] = ModelMetadata(**meta_dict)

            logger.info(f"Loaded {len(self.models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                key: meta.dict()
                for key, meta in self.models.items()
            }

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Saved registry to disk")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
```

### Model Conversion Script
```python
# scripts/convert_finetuned_to_onnx.py
"""
Convert fine-tuned PyTorch embedding models to ONNX format.

Usage:
    python scripts/convert_finetuned_to_onnx.py \\
        --model-path ./my_finetuned_model.pt \\
        --output-path ./my_finetuned_model.onnx \\
        --validate
"""

import argparse
import logging
from pathlib import Path
import torch
import onnx
from onnxruntime import InferenceSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_finetuned_to_onnx(
    model_path: Path,
    output_path: Path,
    input_shape: tuple = (1, 512),
    opset_version: int = 14
) -> Path:
    """
    Convert fine-tuned PyTorch model to ONNX.

    Args:
        model_path: Path to PyTorch model
        output_path: Path for ONNX output
        input_shape: Input shape (batch_size, seq_length)
        opset_version: ONNX opset version

    Returns:
        Path to ONNX model
    """
    logger.info(f"Converting {model_path} to ONNX...")

    # Load PyTorch model
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Create dummy input
    dummy_input = {
        "input_ids": torch.randint(0, 30522, input_shape, dtype=torch.long),
        "attention_mask": torch.ones(input_shape, dtype=torch.long)
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    logger.info(f"Model exported to {output_path}")

    return output_path


def validate_onnx_model(
    onnx_path: Path,
    pytorch_path: Path,
    test_input: dict = None
) -> bool:
    """
    Validate ONNX model against PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_path: Path to PyTorch model
        test_input: Test input (optional)

    Returns:
        True if validation passes
    """
    logger.info("Validating ONNX model...")

    # Load models
    pytorch_model = torch.load(pytorch_path, map_location="cpu")
    pytorch_model.eval()

    onnx_session = InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    # Create test input
    if test_input is None:
        test_input = {
            "input_ids": torch.randint(0, 30522, (1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        }

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(**test_input)

    # ONNX inference
    onnx_input = {
        "input_ids": test_input["input_ids"].numpy(),
        "attention_mask": test_input["attention_mask"].numpy()
    }
    onnx_output = onnx_session.run(None, onnx_input)[0]

    # Compare outputs
    pytorch_embeddings = pytorch_output.last_hidden_state.numpy()
    max_diff = abs(pytorch_embeddings - onnx_output).max()

    logger.info(f"Max difference: {max_diff}")

    if max_diff < 1e-5:
        logger.info("✓ Validation passed!")
        return True
    else:
        logger.warning(f"⚠ Validation warning: max diff {max_diff} > 1e-5")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned models to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to PyTorch model")
    parser.add_argument("--output-path", required=True, help="Output path for ONNX model")
    parser.add_argument("--validate", action="store_true", help="Validate conversion")
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    # Convert
    onnx_path = convert_finetuned_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=args.opset_version
    )

    # Validate if requested
    if args.validate:
        validate_onnx_model(onnx_path, model_path)


if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/services/embeddings/fine_tuned/test_model_registry.py
import pytest
from pathlib import Path
from rag_factory.services.embeddings.fine_tuned.model_registry import (
    ONNXModelRegistry,
    ModelMetadata
)


class TestONNXModelRegistry:
    """Test ONNX model registry."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create temporary registry."""
        return ONNXModelRegistry(registry_dir=tmp_path)

    @pytest.fixture
    def mock_onnx_model(self, tmp_path):
        """Create mock ONNX model file."""
        model_path = tmp_path / "test_model.onnx"
        model_path.write_text("mock onnx model")
        return model_path

    def test_register_model(self, registry, mock_onnx_model):
        """Test model registration."""
        metadata = registry.register_model(
            model_id="test-model",
            model_path=mock_onnx_model,
            version="1.0.0",
            format="onnx",
            description="Test model",
            validate=False  # Skip validation for mock
        )

        assert metadata.model_id == "test-model"
        assert metadata.version == "1.0.0"
        assert metadata.format == "onnx"

    def test_get_model(self, registry, mock_onnx_model):
        """Test model retrieval."""
        registry.register_model(
            model_id="test-model",
            model_path=mock_onnx_model,
            version="1.0.0",
            format="onnx",
            validate=False
        )

        model_path, metadata = registry.get_model("test-model", version="1.0.0")

        assert model_path.exists()
        assert metadata.model_id == "test-model"

    def test_list_models(self, registry, mock_onnx_model):
        """Test listing models."""
        registry.register_model(
            model_id="model1",
            model_path=mock_onnx_model,
            version="1.0.0",
            format="onnx",
            validate=False
        )

        models = registry.list_models(format_filter="onnx")

        assert len(models) == 1
        assert models[0].model_id == "model1"

    def test_delete_model(self, registry, mock_onnx_model):
        """Test model deletion."""
        registry.register_model(
            model_id="test-model",
            model_path=mock_onnx_model,
            version="1.0.0",
            format="onnx",
            validate=False
        )

        registry.delete_model("test-model", "1.0.0")

        with pytest.raises(ValueError):
            registry.get_model("test-model", version="1.0.0")
```

---

## Implementation Plan

### Phase 1: Model Registry (Days 1-2)
1. Update `CustomModelLoader` for ONNX
2. Implement ONNX model registry
3. Add metadata management
4. Implement versioning

### Phase 2: Conversion Tools (Days 2-3)
1. Create conversion script
2. Implement validation
3. Test with fine-tuned models
4. Document process

### Phase 3: A/B Testing Updates (Day 4)
1. Update A/B testing for ONNX
2. Test ONNX vs PyTorch comparison
3. Validate metrics tracking

### Phase 4: Testing and Documentation (Day 5)
1. Write unit tests
2. Write integration tests
3. Create documentation
4. Provide examples

---

## Success Metrics

- [ ] ONNX models supported in registry
- [ ] Conversion tools working
- [ ] A/B testing works with ONNX
- [ ] Quality maintained (>99% accuracy)
- [ ] All tests passing
- [ ] Documentation complete

---

## Dependencies

**Blocked by:** Story 10.1 (ONNX embeddings)
**Blocks:** None
**Related:** Epic 4 (fine-tuned embeddings)
