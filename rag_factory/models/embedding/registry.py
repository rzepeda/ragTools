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
    with JSON-based persistence.
    
    Attributes:
        registry_path: Path to registry directory
        metadata_file: Path to metadata JSON file
        models: Dictionary of model_id -> EmbeddingModelMetadata
    """

    def __init__(self, registry_path: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            registry_path: Path to registry directory (created if doesn't exist)
        """
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
                model_id: metadata.model_dump()
                for model_id, metadata in self.models.items()
            }
            json.dump(data, f, indent=2, default=str)

    def register_model(self, metadata: EmbeddingModelMetadata) -> None:
        """Register a new model.
        
        Args:
            metadata: Model metadata to register
        """
        logger.info(f"Registering model: {metadata.model_id}")

        if metadata.model_id in self.models:
            logger.warning(f"Model {metadata.model_id} already exists, overwriting")

        self.models[metadata.model_id] = metadata
        self._save_registry()

    def get_model(self, model_id: str) -> Optional[EmbeddingModelMetadata]:
        """Get model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        return self.models.get(model_id)

    def list_models(
        self,
        domain: Optional[str] = None,
        format: Optional[ModelFormat] = None,
        tags: Optional[List[str]] = None
    ) -> List[EmbeddingModelMetadata]:
        """List models with optional filtering.
        
        Args:
            domain: Filter by domain
            format: Filter by model format
            tags: Filter by tags (any match)
            
        Returns:
            List of matching model metadata
        """
        models = list(self.models.values())

        if domain:
            models = [m for m in models if m.domain == domain]

        if format:
            models = [m for m in models if m.format == format]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found
        """
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
        """Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of metric_name -> value
            
        Raises:
            ValueError: If model not found
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        for key, value in metrics.items():
            if hasattr(model, key):
                setattr(model, key, value)

        self._save_registry()
        logger.info(f"Updated metrics for model {model_id}")

    def search_models(self, query: str) -> List[EmbeddingModelMetadata]:
        """Search models by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        for model in self.models.values():
            if (query_lower in model.model_name.lower() or
                (model.description and query_lower in model.description.lower())):
                results.append(model)

        return results
