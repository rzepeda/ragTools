from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from pydantic import BaseModel
import onnxruntime as ort
import shutil

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
