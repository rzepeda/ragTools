"""Local sentence-transformers embedding provider implementation."""

from typing import List, Dict, Any
import logging

try:
    from sentence_transformers import SentenceTransformer
    import torch
    LOCAL_AVAILABLE = True
except ImportError:
    LOCAL_AVAILABLE = False

from ..base import IEmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)


class LocalProvider(IEmbeddingProvider):
    """Local sentence-transformers embedding provider.

    Supports local models from sentence-transformers including:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - paraphrase-MiniLM-L6-v2
    - And many others from HuggingFace
    """

    # Common models with their dimensions
    KNOWN_MODELS = {
        "all-MiniLM-L6-v2": {"dimensions": 384},
        "all-mpnet-base-v2": {"dimensions": 768},
        "paraphrase-MiniLM-L6-v2": {"dimensions": 384},
        "paraphrase-mpnet-base-v2": {"dimensions": 768},
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize local provider.

        Args:
            config: Configuration dictionary with 'model' and optional
                   'device', 'max_batch_size'

        Raises:
            ImportError: If sentence-transformers package is not installed
        """
        if not LOCAL_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        device = config.get("device", "cpu")
        self.max_batch_size = config.get("max_batch_size", 32)

        logger.info(f"Loading local model: {self.model_name} on {device}")

        try:
            self.model = SentenceTransformer(self.model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

        # Get actual dimensions from the model
        self._dimensions = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Loaded {self.model_name} with {self._dimensions} dimensions"
        )

    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using local model.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.max_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Estimate token count (rough estimate)
            token_count = sum(len(text.split()) for text in texts)

            return EmbeddingResult(
                embeddings=embeddings_list,
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=token_count,
                cost=0.0,  # Local models have no cost
                provider="local",
                cached=[False] * len(texts)
            )
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions for the current model.

        Returns:
            Number of dimensions in embedding vectors
        """
        return self._dimensions

    def get_max_batch_size(self) -> int:
        """Get maximum batch size.

        Returns:
            Maximum number of texts that can be embedded in one call
        """
        return self.max_batch_size

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Name of the model being used
        """
        return self.model_name

    def calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count.

        Local models have no cost.

        Args:
            token_count: Number of tokens processed

        Returns:
            Always returns 0.0
        """
        return 0.0
