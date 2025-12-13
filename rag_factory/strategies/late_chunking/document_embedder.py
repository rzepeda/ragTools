"""
Document embedder for late chunking strategy.

This module handles full document embedding with token-level detail extraction
using ONNX Runtime and transformers tokenizer.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from pathlib import Path

from rag_factory.services.utils.onnx_utils import (
    download_onnx_model,
    create_onnx_session,
    get_model_metadata,
    mean_pooling as onnx_mean_pooling,
)

from .models import DocumentEmbedding, TokenEmbedding, LateChunkingConfig

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Embed full documents with token-level detail using ONNX Runtime."""

    def __init__(self, config: LateChunkingConfig):
        self.config = config

        # Load ONNX model
        logger.info(f"Loading ONNX model: {config.model_name}")
        
        if config.model_path:
            self.model_path = Path(config.model_path)
        else:
            self.model_path = download_onnx_model(config.model_name)

        # Create ONNX session
        self.session = create_onnx_session(self.model_path)
        
        # Get model metadata
        metadata = get_model_metadata(self.session)
        self.embedding_dim = metadata.get("embedding_dim", 384)
        
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")

        # Initialize tokenizer from transformers (matches ONNX model)
        try:
            from transformers import AutoTokenizer
            
            # Load tokenizer from the same model directory
            model_dir = self.model_path.parent
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                logger.info(f"Loaded tokenizer from model directory: {model_dir}")
            except Exception:
                # Fallback: try loading from HuggingFace
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                logger.info(f"Loaded tokenizer from HuggingFace: {config.model_name}")
                
        except ImportError:
            raise ImportError(
                "transformers package required for tokenization. "
                "Install with: pip install transformers"
            )
        
        # Get max length from tokenizer or use config
        if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length < 1000000:
            self.max_length = min(self.tokenizer.model_max_length, config.max_document_tokens)
        else:
            self.max_length = config.max_document_tokens

        logger.info(
            f"DocumentEmbedder initialized: model={config.model_name}, "
            f"max_length={self.max_length}"
        )

    def embed_document(
        self,
        text: str,
        document_id: str
    ) -> DocumentEmbedding:
        """
        Embed full document and extract token-level embeddings.

        Args:
            text: Document text
            document_id: Unique document ID

        Returns:
            DocumentEmbedding with token-level details
        """
        logger.info(f"Embedding document: {document_id}")

        # Tokenize with transformers tokenizer
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            add_special_tokens=True
        )
        
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        
        token_ids = input_ids[0].tolist()  # Convert to list for processing
        
        if len(token_ids) < len(self.tokenizer.encode(text, add_special_tokens=True)):
            logger.warning(
                f"Document truncated to {len(token_ids)} tokens (max: {self.max_length})"
            )

        # Run ONNX inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        # Extract token-level embeddings
        # Output shape: [batch_size, seq_length, embedding_dim]
        token_embeddings_array = outputs[0][0]  # Remove batch dimension

        # Document-level embedding (mean pooling)
        full_embedding = self._mean_pooling(
            token_embeddings_array,
            attention_mask[0]
        )

        # Convert token embeddings to TokenEmbedding objects
        token_embeddings = []
        
        # Decode tokens individually for mapping
        current_pos = 0
        for i, token_id in enumerate(token_ids):
            # Decode individual token
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            # Find token in original text (approximate)
            start_char = text.find(token_str, current_pos) if token_str.strip() else current_pos
            if start_char == -1:
                # Token not found exactly, use approximate position
                start_char = current_pos
            end_char = start_char + len(token_str)
            current_pos = end_char

            token_emb = TokenEmbedding(
                token=token_str,
                token_id=int(token_id),
                start_char=start_char,
                end_char=end_char,
                embedding=token_embeddings_array[i].tolist(),
                position=i
            )
            token_embeddings.append(token_emb)

        # Create DocumentEmbedding
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            text=text,
            full_embedding=full_embedding.tolist(),
            token_embeddings=token_embeddings,
            model_name=self.config.model_name,
            token_count=len(token_embeddings),
            embedding_dim=self.embedding_dim
        )

        logger.info(
            f"Embedded document {document_id}: {len(token_embeddings)} tokens, "
            f"dim {doc_embedding.embedding_dim}"
        )

        return doc_embedding

    def _mean_pooling(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Mean pooling to get document-level embedding.
        
        Args:
            token_embeddings: Token-level embeddings [seq_length, embedding_dim]
            attention_mask: Attention mask [seq_length]
            
        Returns:
            Document embedding [embedding_dim]
        """
        # Expand attention mask to match embedding dimensions
        attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        attention_mask_expanded = attention_mask_expanded.astype(np.float32)

        # Sum embeddings, weighted by attention mask
        sum_embeddings = np.sum(token_embeddings * attention_mask_expanded, axis=0)

        # Sum attention mask to get counts
        sum_mask = np.sum(attention_mask_expanded, axis=0)

        # Avoid division by zero
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)

        # Mean pooling
        return sum_embeddings / sum_mask

    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to token strings.

        Args:
            token_ids: List of token IDs

        Returns:
            List of token strings
        """
        tokens = []
        for token_id in token_ids:
            try:
                token_str = self.tokenizer.decode([token_id])
                tokens.append(token_str)
            except Exception:
                tokens.append(f"<unk_{token_id}>")

        return tokens

    def chunk_embeddings(
        self,
        token_embeddings: np.ndarray,
        tokens: List[str],
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Tuple[np.ndarray, List[str], int, int]]:
        """
        Chunk token embeddings into smaller pieces.

        Args:
            token_embeddings: Token-level embeddings [num_tokens, dim]
            tokens: Token strings
            chunk_size: Tokens per chunk
            overlap: Overlapping tokens

        Returns:
            List of (chunk_embeddings, chunk_tokens, start_idx, end_idx)
        """
        num_tokens = len(token_embeddings)
        chunks = []

        start = 0
        while start < num_tokens:
            end = min(start + chunk_size, num_tokens)

            chunk_emb = token_embeddings[start:end]
            chunk_tok = tokens[start:end]

            chunks.append((chunk_emb, chunk_tok, start, end))

            # If we've reached the end, we're done
            if end >= num_tokens:
                break

            # Move to next chunk with overlap
            start = end - overlap
            
            # Ensure we make progress (avoid infinite loop if overlap >= chunk_size)
            if start <= chunks[-1][2]:  # If new start <= previous start
                start = chunks[-1][2] + 1

        return chunks

    def pool_embeddings(
        self,
        token_embeddings: np.ndarray,
        method: str = "mean"
    ) -> np.ndarray:
        """
        Pool token embeddings to document embedding.

        Args:
            token_embeddings: Token-level embeddings [num_tokens, dim]
            method: Pooling method ("mean", "max", "first")

        Returns:
            Document embedding [dim]
        """
        if method == "mean":
            return np.mean(token_embeddings, axis=0)
        elif method == "max":
            return np.max(token_embeddings, axis=0)
        elif method == "first":
            return token_embeddings[0]
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def embed_documents_batch(
        self,
        documents: List[Dict[str, str]]
    ) -> List[DocumentEmbedding]:
        """
        Embed multiple documents in batch.

        Args:
            documents: List of {"text": str, "document_id": str}

        Returns:
            List of DocumentEmbedding objects
        """
        return [
            self.embed_document(doc["text"], doc["document_id"])
            for doc in documents
        ]
