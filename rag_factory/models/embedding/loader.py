"""Custom model loader for embedding models."""

from typing import List, Dict, Any, Optional, Union
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from rag_factory.models.embedding.models import ModelConfig, ModelFormat, PoolingStrategy

logger = logging.getLogger(__name__)


class CustomModelLoader:
    """Load custom embedding models.
    
    Supports multiple model formats:
    - Sentence-Transformers
    - Hugging Face Transformers
    - ONNX (placeholder for future implementation)
    
    Models are cached after first load for efficiency.
    """

    def __init__(self):
        """Initialize model loader with empty cache."""
        self.loaded_models: Dict[str, Any] = {}

    def load_model(self, config: ModelConfig) -> Any:
        """Load embedding model based on configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model format is unsupported
            ImportError: If required dependencies are not installed
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

    def _load_sentence_transformer(self, config: ModelConfig) -> Any:
        """Load Sentence-Transformers model.
        
        Args:
            config: Model configuration
            
        Returns:
            SentenceTransformer model
            
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SENTENCE_TRANSFORMERS format. "
                "Install with: pip install sentence-transformers"
            )

        model = SentenceTransformer(config.model_path, device=config.device)

        if config.use_fp16 and config.device != "cpu":
            model = model.half()

        return model

    def _load_huggingface(self, config: ModelConfig) -> Dict[str, Any]:
        """Load Hugging Face model.
        
        Args:
            config: Model configuration
            
        Returns:
            Dictionary with model, tokenizer, and config
            
        Raises:
            ImportError: If transformers is not installed
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HUGGINGFACE format. "
                "Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModel.from_pretrained(config.model_path)

        model.to(config.device)
        model.eval()

        if config.use_fp16 and config.device != "cpu":
            model = model.half()

        return {"model": model, "tokenizer": tokenizer, "config": config}

    def _load_onnx(self, config: ModelConfig) -> Any:
        """Load ONNX model.
        
        Args:
            config: Model configuration
            
        Returns:
            ONNX InferenceSession
            
        Raises:
            ImportError: If onnxruntime is not installed
            NotImplementedError: ONNX loading is not yet fully implemented
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX format. "
                "Install with: pip install onnxruntime"
            )

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
        """Generate embeddings for texts using loaded model.
        
        Args:
            texts: List of texts to embed
            model: Loaded model
            config: Model configuration
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If model format is unsupported
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
        """Generate embeddings using Hugging Face model.
        
        Args:
            texts: List of texts to embed
            model_dict: Dictionary with model and tokenizer
            config: Model configuration
            
        Returns:
            List of embedding vectors
        """
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
        session: Any,
        config: ModelConfig
    ) -> List[List[float]]:
        """Generate embeddings using ONNX model.
        
        Args:
            texts: List of texts to embed
            session: ONNX InferenceSession
            config: Model configuration
            
        Returns:
            List of embedding vectors
            
        Raises:
            NotImplementedError: ONNX embedding is not yet fully implemented
        """
        raise NotImplementedError(
            "ONNX embedding not yet implemented. "
            "Requires tokenizer and proper input preparation."
        )

    def _pool_embeddings(
        self,
        token_embeddings: Any,  # torch.Tensor when available
        attention_mask: Any,  # torch.Tensor when available
        strategy: PoolingStrategy
    ) -> Any:  # torch.Tensor when available
        """Pool token embeddings into single embedding.
        
        Args:
            token_embeddings: Token-level embeddings [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]
            strategy: Pooling strategy to use
            
        Returns:
            Pooled embeddings [batch, hidden_dim]
            
        Raises:
            ValueError: If pooling strategy is unknown
            ImportError: If torch is not available
        """
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required for embedding pooling")
            
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
            token_embeddings = token_embeddings.clone()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
