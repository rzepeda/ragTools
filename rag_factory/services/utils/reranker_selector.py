"""
Reranker selector utility for automatic reranker selection.

This module provides utilities to automatically select the best available
reranker based on installed dependencies and configuration.
"""

from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class RerankerSelector:
    """
    Automatically select the best available reranker.
    
    Priority order:
    1. Cohere (if API key available)
    2. Cosine similarity (always available)
    3. PyTorch-based rerankers (if torch installed)
    
    Example:
        >>> from rag_factory.services.embedding import ONNXEmbeddingProvider
        >>> embedder = ONNXEmbeddingProvider()
        >>> reranker = RerankerSelector.select_reranker(embedder)
        >>> # Returns best available reranker
    """
    
    @staticmethod
    def select_reranker(
        embedding_provider: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Select the best available reranker.
        
        Args:
            embedding_provider: Embedding provider for cosine reranker
            config: Optional configuration dictionary with keys:
                - reranker_type: Manual selection ("cohere", "cosine", "bge", "cross-encoder")
                - cohere_api_key: Cohere API key (overrides env var)
                - cohere_model: Cohere model name
                - similarity_metric: Metric for cosine reranker ("cosine", "dot", "euclidean")
                - normalize: Whether to normalize embeddings (bool)
                - bge_model: BGE model name
                - cross_encoder_model: Cross-encoder model name
                
        Returns:
            Reranker instance
            
        Raises:
            ImportError: If manually selected reranker is not available
            ValueError: If reranker_type is unknown
        """
        config = config or {}
        
        # Check for manual selection
        if "reranker_type" in config:
            return RerankerSelector._create_reranker(
                config["reranker_type"],
                embedding_provider,
                config
            )
        
        # Auto-select based on availability
        # 1. Try Cohere (if API key available)
        if RerankerSelector._is_cohere_available(config):
            logger.info("Auto-selected Cohere reranker (API-based, high quality)")
            from rag_factory.strategies.reranking.cohere_reranker import CohereReranker
            from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel
            
            rerank_config = RerankConfig(
                model=RerankerModel.COHERE,
                model_name=config.get("cohere_model", "rerank-english-v3.0"),
                model_config={
                    "api_key": config.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
                }
            )
            return CohereReranker(rerank_config)
        
        # 2. Fall back to Cosine similarity (always available)
        logger.info("Auto-selected Cosine similarity reranker (lightweight, no API required)")
        from rag_factory.strategies.reranking.cosine_reranker import CosineReranker
        from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel
        
        rerank_config = RerankConfig(
            model=RerankerModel.COSINE,
            model_config={
                "embedding_provider": embedding_provider,
                "metric": config.get("similarity_metric", "cosine"),
                "normalize": config.get("normalize", True)
            }
        )
        return CosineReranker(
            rerank_config,
            embedding_provider=embedding_provider,
            metric=config.get("similarity_metric", "cosine"),
            normalize=config.get("normalize", True)
        )
    
    @staticmethod
    def _is_cohere_available(config: Dict[str, Any]) -> bool:
        """
        Check if Cohere API is available.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if Cohere is available, False otherwise
        """
        # Check for API key in config or environment
        api_key = config.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
        if not api_key:
            return False
        
        # Check if cohere package is installed
        try:
            import cohere
            return True
        except ImportError:
            logger.debug("Cohere package not installed")
            return False
    
    @staticmethod
    def _is_torch_available() -> bool:
        """
        Check if PyTorch is available.
        
        Returns:
            True if PyTorch is installed, False otherwise
        """
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _create_reranker(
        reranker_type: str,
        embedding_provider: Any,
        config: Dict[str, Any]
    ) -> Any:
        """
        Create specific reranker type.
        
        Args:
            reranker_type: Type of reranker to create
            embedding_provider: Embedding provider for cosine reranker
            config: Configuration dictionary
            
        Returns:
            Reranker instance
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If reranker_type is unknown
        """
        from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel
        
        if reranker_type == "cohere":
            if not RerankerSelector._is_cohere_available(config):
                raise ImportError(
                    "Cohere reranker requires cohere package and API key. "
                    "Install with: pip install cohere tenacity\n"
                    "Set COHERE_API_KEY environment variable or pass in config."
                )
            from rag_factory.strategies.reranking.cohere_reranker import CohereReranker
            
            rerank_config = RerankConfig(
                model=RerankerModel.COHERE,
                model_name=config.get("cohere_model", "rerank-english-v3.0"),
                model_config={
                    "api_key": config.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
                }
            )
            return CohereReranker(rerank_config)
        
        elif reranker_type == "cosine":
            from rag_factory.strategies.reranking.cosine_reranker import CosineReranker
            
            rerank_config = RerankConfig(
                model=RerankerModel.COSINE,
                model_config={
                    "embedding_provider": embedding_provider,
                    "metric": config.get("similarity_metric", "cosine"),
                    "normalize": config.get("normalize", True)
                }
            )
            return CosineReranker(
                rerank_config,
                embedding_provider=embedding_provider,
                metric=config.get("similarity_metric", "cosine"),
                normalize=config.get("normalize", True)
            )
        
        elif reranker_type == "bge":
            if not RerankerSelector._is_torch_available():
                raise ImportError(
                    "BGE reranker requires PyTorch and transformers. "
                    "Install with: pip install torch transformers\n"
                    "Alternatively, use 'cohere' or 'cosine' reranker for lightweight deployment."
                )
            from rag_factory.strategies.reranking.bge_reranker import BGEReranker
            
            rerank_config = RerankConfig(
                model=RerankerModel.BGE,
                model_name=config.get("bge_model", "BAAI/bge-reranker-base")
            )
            return BGEReranker(rerank_config)
        
        elif reranker_type == "cross-encoder":
            if not RerankerSelector._is_torch_available():
                raise ImportError(
                    "Cross-Encoder reranker requires PyTorch and sentence-transformers. "
                    "Install with: pip install torch sentence-transformers\n"
                    "Alternatively, use 'cohere' or 'cosine' reranker for lightweight deployment."
                )
            from rag_factory.strategies.reranking.cross_encoder_reranker import CrossEncoderReranker
            
            rerank_config = RerankConfig(
                model=RerankerModel.CROSS_ENCODER,
                model_name=config.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
            return CrossEncoderReranker(rerank_config)
        
        else:
            raise ValueError(
                f"Unknown reranker type: {reranker_type}. "
                f"Supported types: 'cohere', 'cosine', 'bge', 'cross-encoder'"
            )
    
    @staticmethod
    def get_available_rerankers() -> Dict[str, bool]:
        """
        Get dictionary of available rerankers.
        
        Returns:
            Dictionary mapping reranker names to availability status
            
        Example:
            >>> available = RerankerSelector.get_available_rerankers()
            >>> print(available)
            {'cohere': True, 'cosine': True, 'bge': False, 'cross-encoder': False}
        """
        return {
            "cohere": RerankerSelector._is_cohere_available({}),
            "cosine": True,  # Always available (uses numpy)
            "bge": RerankerSelector._is_torch_available(),
            "cross-encoder": RerankerSelector._is_torch_available()
        }
