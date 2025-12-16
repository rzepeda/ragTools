"""Custom model loader for embedding models."""

from typing import List, Dict, Any, Optional, Union
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

from rag_factory.models.embedding.models import ModelConfig, ModelFormat, PoolingStrategy

logger = logging.getLogger(__name__)


class CustomModelLoader:
    """Load custom embedding models.
    
    Supports multiple model formats:
    - Sentence-Transformers
    - Hugging Face Transformers
    - ONNX (with tiktoken tokenization)
    
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

        # Prioritize ONNX if requested or if format is ONNX
        if config.model_format == ModelFormat.ONNX or config.use_onnx:
            try:
                model = self._load_onnx(config)
                # Update format in config if we successfully loaded ONNX when it wasn't explicitly set
                if config.model_format != ModelFormat.ONNX:
                    logger.info("Successfully loaded ONNX model, updating config format")
                    config.model_format = ModelFormat.ONNX
            except Exception as e:
                if config.model_format == ModelFormat.ONNX:
                    raise e
                logger.warning(f"Failed to load ONNX model, falling back to {config.model_format}: {e}")
                model = self._load_fallback(config)
        else:
            model = self._load_fallback(config)

        # Cache model
        self.loaded_models[cache_key] = model

        logger.info(f"Model loaded successfully: {config.model_format.value}")
        return model

    def _load_fallback(self, config: ModelConfig) -> Any:
        """Load using fallback formats (Sentence-Transformers or Hugging Face)."""
        if config.model_format == ModelFormat.SENTENCE_TRANSFORMERS:
            return self._load_sentence_transformer(config)
        elif config.model_format == ModelFormat.HUGGINGFACE:
            return self._load_huggingface(config)
        else:
            raise ValueError(f"Unsupported model format: {config.model_format}")

    def _load_sentence_transformer(self, config: ModelConfig) -> Any:
        """Load Sentence-Transformers model."""
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
        """Load Hugging Face model."""
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
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX format. "
                "Install with: pip install onnxruntime"
            )

        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for ONNX embedding. "
                "Install with: pip install tiktoken"
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
        """Generate embeddings for texts using loaded model."""
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
        session: Any,
        config: ModelConfig
    ) -> List[List[float]]:
        """Generate embeddings using ONNX model with tiktoken."""
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for ONNX embedding")

        # Get tokenizer
        tokenizer_name = config.tokenizer_name or "cl100k_base"
        try:
            encoding = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            # Fallback to cl100k_base if name is invalid
            encoding = tiktoken.get_encoding("cl100k_base")

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i:i + config.batch_size]
            
            # Tokenize
            batch_input_ids = []
            batch_attention_mask = []
            max_len = 0
            
            for text in batch:
                tokens = encoding.encode(text)
                # Truncate if needed (assuming 512 limit for standard BERT-like models, 
                # but tiktoken models might handle more. Using 512 as safe default or config value)
                max_seq_len = config.additional_config.get("max_seq_length", 512)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                
                batch_input_ids.append(tokens)
                max_len = max(max_len, len(tokens))
            
            # Pad
            padded_input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            padded_attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            
            for j, tokens in enumerate(batch_input_ids):
                padded_input_ids[j, :len(tokens)] = tokens
                padded_attention_mask[j, :len(tokens)] = 1
                
            # ONNX Inference
            onnx_inputs = {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask
            }
            
            # Some models might need token_type_ids
            input_names = [node.name for node in session.get_inputs()]
            if "token_type_ids" in input_names:
                onnx_inputs["token_type_ids"] = np.zeros_like(padded_input_ids)
                
            outputs = session.run(None, onnx_inputs)
            last_hidden_state = outputs[0]
            
            # Pool embeddings (using numpy)
            batch_embeddings = self._pool_embeddings_numpy(
                last_hidden_state,
                padded_attention_mask,
                config.pooling_strategy
            )
            
            # Normalize
            if config.normalize_embeddings:
                norm = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norm, 1e-9)
                
            embeddings.extend(batch_embeddings.tolist())
            
        return embeddings

    def _pool_embeddings(
        self,
        token_embeddings: Any,
        attention_mask: Any,
        strategy: PoolingStrategy
    ) -> Any:
        """Pool token embeddings using PyTorch."""
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required for PyTorch embedding pooling")
            
        if strategy == PoolingStrategy.MEAN:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif strategy == PoolingStrategy.CLS:
            return token_embeddings[:, 0, :]

        elif strategy == PoolingStrategy.MAX:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings = token_embeddings.clone()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def _pool_embeddings_numpy(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
        strategy: PoolingStrategy
    ) -> np.ndarray:
        """Pool token embeddings using NumPy."""
        if strategy == PoolingStrategy.MEAN:
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
            return sum_embeddings / sum_mask

        elif strategy == PoolingStrategy.CLS:
            return token_embeddings[:, 0, :]

        elif strategy == PoolingStrategy.MAX:
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
            # Set masked values to very small number
            token_embeddings_masked = token_embeddings.copy()
            token_embeddings_masked[input_mask_expanded == 0] = -1e9
            return np.max(token_embeddings_masked, axis=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
