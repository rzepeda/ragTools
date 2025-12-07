"""
Document embedder for late chunking strategy.

This module handles full document embedding with token-level detail extraction.
"""

from typing import List, Dict, Any
import logging
import torch
from transformers import AutoTokenizer, AutoModel

from .models import DocumentEmbedding, TokenEmbedding, LateChunkingConfig

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Embed full documents with token-level detail."""

    def __init__(self, config: LateChunkingConfig):
        self.config = config
        self.device = config.device

        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = min(
            config.max_document_tokens,
            self.tokenizer.model_max_length
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

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=True
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0]

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Extract embeddings
        # Use last hidden state for token embeddings
        token_embeddings_tensor = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

        # Document-level embedding (mean pooling)
        full_embedding = self._mean_pooling(
            token_embeddings_tensor,
            attention_mask[0]
        )

        # Convert token embeddings to TokenEmbedding objects
        token_embeddings = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        for i, (token, offset) in enumerate(zip(tokens, offset_mapping)):
            if i == 0 or i == len(tokens) - 1:  # Skip special tokens
                continue

            start_char, end_char = offset.tolist()

            token_emb = TokenEmbedding(
                token=token,
                token_id=int(input_ids[0][i]),
                start_char=start_char,
                end_char=end_char,
                embedding=token_embeddings_tensor[i].cpu().tolist(),
                position=i
            )
            token_embeddings.append(token_emb)

        # Create DocumentEmbedding
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            text=text,
            full_embedding=full_embedding.cpu().tolist(),
            token_embeddings=token_embeddings,
            model_name=self.config.model_name,
            token_count=len(token_embeddings),
            embedding_dim=token_embeddings_tensor.shape[1]
        )

        logger.info(
            f"Embedded document {document_id}: {len(token_embeddings)} tokens, "
            f"dim {doc_embedding.embedding_dim}"
        )

        return doc_embedding

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling to get document-level embedding."""
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=0)

        # Divide by number of tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=0), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        return mean_embedding

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
