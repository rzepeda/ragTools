"""ONNX-based local embedding provider (90% smaller than PyTorch).

This provider uses ONNX Runtime instead of PyTorch for local embeddings,
reducing dependencies from ~2.5GB to ~200MB while maintaining performance.

NO PYTORCH DEPENDENCIES - Uses pure numpy for all operations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np
import json

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False



from ..base import IEmbeddingProvider, EmbeddingResult
from ...utils.onnx_utils import (
    get_onnx_model_path,
    create_onnx_session,
    validate_onnx_model,
    get_model_metadata,
    mean_pooling,
    normalize_embeddings,
)
from rag_factory.utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)


class ONNXLocalProvider(IEmbeddingProvider):
    """ONNX-optimized local embedding provider.

    This provider offers the same functionality as the sentence-transformers
    provider but with 90% smaller dependencies:
    - PyTorch approach: ~2.5GB
    - ONNX approach: ~200MB

    Supports any sentence-transformers compatible model from HuggingFace:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, ~90MB)
    - sentence-transformers/all-mpnet-base-v2 (768 dim, ~420MB)
    - sentence-transformers/paraphrase-MiniLM-L6-v2 (384 dim)
    - BAAI/bge-small-en-v1.5 (384 dim, state-of-the-art)
    - BAAI/bge-base-en-v1.5 (768 dim)

    Benefits:
    - 90% smaller dependencies
    - Faster inference (ONNX optimized)
    - Same model compatibility
    - Zero API costs
    - NO PyTorch or CUDA required
    """

    # Common models with their dimensions
    KNOWN_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": {"dimensions": 384, "size_mb": 90},
        "Xenova/all-MiniLM-L6-v2": {"dimensions": 384, "size_mb": 90},
        "sentence-transformers/all-mpnet-base-v2": {"dimensions": 768, "size_mb": 420},
        "Xenova/all-mpnet-base-v2": {"dimensions": 768, "size_mb": 420},
        "sentence-transformers/paraphrase-MiniLM-L6-v2": {"dimensions": 384, "size_mb": 90},
        "sentence-transformers/paraphrase-mpnet-base-v2": {"dimensions": 768, "size_mb": 420},
        "BAAI/bge-small-en-v1.5": {"dimensions": 384, "size_mb": 133},
        "BAAI/bge-base-en-v1.5": {"dimensions": 768, "size_mb": 438},
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize ONNX local provider.

        Args:
            config: Configuration dictionary with:
                - model: Model name (default: from EMBEDDING_MODEL_NAME env or Xenova/all-mpnet-base-v2)
                - model_path: Path to local ONNX model (optional, from EMBEDDING_MODEL_PATH env)
                - max_batch_size: Maximum batch size (default: 32)
                - cache_dir: Directory for model cache (optional)
                - num_threads: Number of CPU threads (optional)
                - max_length: Maximum sequence length (default: 512)

        Raises:
            ImportError: If required packages not installed
        """
        import os
        
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime not installed. Install with:\n"
                "  pip install onnx>=1.15.0 onnxruntime>=1.16.3\n"
                "Total size: ~215MB (vs ~2.5GB for PyTorch)"
            )



        # Get model name from config or environment variable
        # Default to Xenova model (has pre-converted ONNX files)
        default_model = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")
        self.model_name = config.get("model", default_model)
        
        self.max_batch_size = config.get("max_batch_size", 32)
        self.max_length = config.get("max_length", 512)
        
        # Get cache directory from config or environment
        cache_dir = config.get("cache_dir") or os.getenv("EMBEDDING_MODEL_PATH")
        num_threads = config.get("num_threads", None)
        model_path = config.get("model_path", None)

        logger.info(f"Loading ONNX model: {self.model_name}")

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = get_onnx_model_path(
                self.model_name,
                cache_dir=Path(cache_dir) if cache_dir else None
            )

        # Create ONNX session
        self.session = create_onnx_session(
            self.model_path,
            num_threads=num_threads
        )

        # Validate model
        validate_onnx_model(
            self.session,
            expected_inputs=["input_ids", "attention_mask"],
        )

        # Get model metadata
        metadata = get_model_metadata(self.session)
        
        # Determine dimensions
        if self.model_name in self.KNOWN_MODELS:
            self._dimensions = self.KNOWN_MODELS[self.model_name]["dimensions"]
        elif "embedding_dim" in metadata:
            self._dimensions = metadata["embedding_dim"]
        else:
            # Try to infer from output shape
            output_shape = self.session.get_outputs()[0].shape
            if len(output_shape) >= 2:
                self._dimensions = output_shape[-1]
            else:
                raise ValueError(
                    f"Could not determine embedding dimension for {self.model_name}"
                )

        # Initialize tokenizer from model directory
        # Use lightweight tokenizers library (no PyTorch dependency)
        # The model_path points to onnx/model.onnx, so we need to go up to the model root
        if "onnx" in str(self.model_path):
            # Model is in onnx/ subdirectory, go up two levels
            model_dir = self.model_path.parent.parent
        else:
            model_dir = self.model_path.parent
            
        self.hf_tokenizer = None
        
        try:
            from tokenizers import Tokenizer as HFTokenizer
            # Look for tokenizer.json in model directory
            tokenizer_path = model_dir / "tokenizer.json"
            if tokenizer_path.exists():
                self.hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            else:
                logger.warning(f"No tokenizer.json found in {model_dir}")
                raise FileNotFoundError("tokenizer.json not found")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            logger.warning("Falling back to basic tokenizer (may not work correctly)")
            self.tokenizer = Tokenizer(encoding_name="cl100k_base", use_fallback=True)

        logger.info(
            f"Loaded ONNX model {self.model_name} with {self._dimensions} dimensions"
        )

    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using ONNX model.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=0,
                cost=0.0,
                provider="onnx-local",
                cached=[],
            )

        try:
            # Tokenize texts
            input_ids, attention_mask = self._tokenize(texts)

            # Prepare inputs for ONNX model
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            # Check if model requires token_type_ids (BERT-based models like MiniLM)
            input_names = [inp.name for inp in self.session.get_inputs()]
            if "token_type_ids" in input_names:
                # Create token_type_ids (all zeros for single sequence)
                token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
                inputs["token_type_ids"] = token_type_ids

            # Run inference with ONNX
            outputs = self.session.run(None, inputs)  # Get all outputs

            # Extract embeddings (first output is usually last_hidden_state)
            token_embeddings = outputs[0]

            # Apply mean pooling
            embeddings = mean_pooling(token_embeddings, attention_mask)

            # Normalize embeddings (L2 normalization)
            embeddings = normalize_embeddings(embeddings)

            # Convert to list of lists
            embeddings_list = embeddings.tolist()

            # Estimate token count
            token_count = int(np.sum(attention_mask))

            return EmbeddingResult(
                embeddings=embeddings_list,
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=token_count,
                cost=0.0,  # Local models have no cost
                provider="onnx-local",
                cached=[False] * len(texts),
            )
        except Exception as e:
            logger.error(f"ONNX embedding error: {e}")
            raise

    def _tokenize(self, texts: List[str]) -> tuple:
        """Tokenize texts for model input.

        Args:
            texts: List of texts to tokenize

        Returns:
            Tuple of (input_ids, attention_mask) as numpy arrays
        """
        if self.hf_tokenizer is not None:
            # Use tokenizers library (lightweight, no PyTorch)
            input_ids = []
            attention_mask = []
            
            for text in texts:
                # Encode with the tokenizer
                encoding = self.hf_tokenizer.encode(text)
                tokens = encoding.ids
                
                # Truncate if needed
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                # Pad to max_length
                padding_length = self.max_length - len(tokens)
                padded_tokens = tokens + [0] * padding_length
                mask = [1] * len(tokens) + [0] * padding_length
                
                input_ids.append(padded_tokens)
                attention_mask.append(mask)
            
            return (
                np.array(input_ids, dtype=np.int64),
                np.array(attention_mask, dtype=np.int64)
            )
        else:
            # Fallback to basic tokenizer (may not work correctly)
            input_ids = []
            attention_mask = []

            for text in texts:
                # Encode text using our tokenizer
                tokens = self.tokenizer.encode(text)
                
                # Truncate if needed
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                # Pad to max_length
                padding_length = self.max_length - len(tokens)
                padded_tokens = tokens + [0] * padding_length
                mask = [1] * len(tokens) + [0] * padding_length
                
                input_ids.append(padded_tokens)
                attention_mask.append(mask)

            return (
                np.array(input_ids, dtype=np.int64),
                np.array(attention_mask, dtype=np.int64)
            )

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
