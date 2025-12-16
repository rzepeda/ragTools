"""Model registry for managing embedding models."""

from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
from rag_factory.models.embedding.models import EmbeddingModelMetadata, ModelFormat

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for embedding models.
    
    Provides centralized storage and management of embedding model metadata
    with JSON-based persistence. Supports versioning and validation.
    
    Attributes:
        registry_path: Path to registry directory
        metadata_file: Path to metadata JSON file
        models: Dictionary of model_id -> version -> EmbeddingModelMetadata
    """

    def __init__(self, registry_path: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            registry_path: Path to registry directory (created if doesn't exist)
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "models.json"
        self.models: Dict[str, Dict[str, EmbeddingModelMetadata]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    
                # Handle both legacy (flat) and new (nested) formats
                for key, value in data.items():
                    if isinstance(value, dict) and "model_id" in value:
                        # Legacy format: key is model_id, value is metadata
                        metadata = EmbeddingModelMetadata(**value)
                        if metadata.model_id not in self.models:
                            self.models[metadata.model_id] = {}
                        self.models[metadata.model_id][metadata.version] = metadata
                    else:
                        # New format: key is model_id, value is dict of versions
                        model_id = key
                        self.models[model_id] = {}
                        for version, meta_dict in value.items():
                            self.models[model_id][version] = EmbeddingModelMetadata(**meta_dict)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models = {}
        else:
            self.models = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {
                model_id: {
                    version: meta.model_dump()
                    for version, meta in versions.items()
                }
                for model_id, versions in self.models.items()
            }
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(self, metadata: EmbeddingModelMetadata, validate: bool = True) -> None:
        """Register a new model version.
        
        Args:
            metadata: Model metadata to register
            validate: Whether to validate the model before registering
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Registering model: {metadata.model_id} version {metadata.version}")

        if validate:
            self._validate_model(metadata)

        if metadata.model_id not in self.models:
            self.models[metadata.model_id] = {}

        if metadata.version in self.models[metadata.model_id]:
            logger.warning(f"Model {metadata.model_id} version {metadata.version} already exists, overwriting")

        self.models[metadata.model_id][metadata.version] = metadata
        self._save_registry()

    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[EmbeddingModelMetadata]:
        """Get model metadata by ID and optional version.
        
        Args:
            model_id: Model identifier
            version: Specific version (None = latest)
            
        Returns:
            Model metadata or None if not found
        """
        if model_id not in self.models:
            return None
            
        versions = self.models[model_id]
        if not versions:
            return None
            
        if version:
            return versions.get(version)
        
        # Get latest version (simple string comparison for now, ideally semver)
        # Assuming semantic versioning, we can sort
        latest_version = sorted(versions.keys(), key=lambda v: [int(p) for p in v.split('.')] if all(p.isdigit() for p in v.split('.')) else v)[-1]
        return versions[latest_version]

    def list_models(
        self,
        domain: Optional[str] = None,
        format: Optional[ModelFormat] = None,
        tags: Optional[List[str]] = None,
        latest_only: bool = True
    ) -> List[EmbeddingModelMetadata]:
        """List models with optional filtering.
        
        Args:
            domain: Filter by domain
            format: Filter by model format
            tags: Filter by tags (any match)
            latest_only: If True, returns only latest version of each model
            
        Returns:
            List of matching model metadata
        """
        results = []
        
        for model_id, versions in self.models.items():
            if latest_only:
                # Get latest
                latest = self.get_model(model_id)
                candidates = [latest] if latest else []
            else:
                candidates = list(versions.values())
                
            for model in candidates:
                if domain and model.domain != domain:
                    continue
                if format and model.format != format:
                    continue
                if tags and not any(tag in model.tags for tag in tags):
                    continue
                results.append(model)

        return results

    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a model or specific version from registry.
        
        Args:
            model_id: Model identifier
            version: Specific version to delete (if None, deletes all versions)
            
        Returns:
            True if deleted, False if not found
        """
        if model_id not in self.models:
            return False

        if version:
            if version in self.models[model_id]:
                del self.models[model_id][version]
                if not self.models[model_id]:
                    del self.models[model_id]
                self._save_registry()
                logger.info(f"Deleted model: {model_id} version {version}")
                return True
            return False
        else:
            del self.models[model_id]
            self._save_registry()
            logger.info(f"Deleted all versions of model: {model_id}")
            return True

    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float],
        version: Optional[str] = None
    ) -> None:
        """Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of metric_name -> value
            version: Specific version (None = latest)
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id} (version {version}) not found")

        for key, value in metrics.items():
            if hasattr(model, key):
                setattr(model, key, value)

        self.models[model_id][model.version] = model
        self._save_registry()
        logger.info(f"Updated metrics for model {model_id} version {model.version}")

    def search_models(self, query: str) -> List[EmbeddingModelMetadata]:
        """Search models by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching models (latest versions)
        """
        query_lower = query.lower()
        results = []

        for model_id in self.models:
            model = self.get_model(model_id)
            if model:
                if (query_lower in model.model_name.lower() or
                    (model.description and query_lower in model.description.lower())):
                    results.append(model)

        return results

    def _validate_model(self, metadata: EmbeddingModelMetadata) -> None:
        """Validate model metadata and compatibility.
        
        Args:
            metadata: Model metadata to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not metadata.model_id or not metadata.version:
            raise ValueError("Model ID and version are required")
            
        # Basic semver check
        import re
        semver_pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(semver_pattern, metadata.version):
            logger.warning(f"Version {metadata.version} does not strictly follow semver (X.Y.Z)")

        # Format specific validation could go here
        if metadata.format == ModelFormat.ONNX and not metadata.onnx_opset_version:
            logger.warning("ONNX model registered without opset version")

    def set_health_status(self, model_id: str, status: str, version: Optional[str] = None) -> None:
        """Set health status for a model.
        
        Args:
            model_id: Model identifier
            status: New status ("healthy", "degraded", "failed")
            version: Specific version (None = latest)
        """
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id} not found")
            
        if status not in ["healthy", "degraded", "failed"]:
            raise ValueError("Invalid health status")
            
        model.health_status = status
        self.models[model_id][model.version] = model
        self._save_registry()

