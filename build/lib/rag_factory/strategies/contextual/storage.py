"""
Storage management for contextualized chunks.

This module provides the ContextualStorageManager class that handles
dual storage of original and contextualized chunks.
"""

from typing import List, Dict, Any
import logging

from .config import ContextualRetrievalConfig

logger = logging.getLogger(__name__)


class ContextualStorageManager:
    """
    Manages storage of original and contextualized chunks.
    
    Implements dual storage pattern where both original and contextualized
    versions of chunks are stored, allowing flexible retrieval options.
    """

    def __init__(self, database_service: Any, config: ContextualRetrievalConfig):
        """
        Initialize storage manager.
        
        Args:
            database_service: Database service for chunk storage
            config: Contextual retrieval configuration
        """
        self.database = database_service
        self.config = config

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store chunks with dual storage (original + contextualized).
        
        Args:
            chunks: List of chunks with context information
        """
        logger.info(f"Storing {len(chunks)} contextualized chunks")
        
        for chunk in chunks:
            self._store_chunk(chunk)

    def _store_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Store a single chunk.
        
        Args:
            chunk: Chunk to store
        """
        chunk_data = {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": chunk.get("document_id"),
        }
        
        # Store original text
        if self.config.store_original:
            chunk_data["original_text"] = chunk.get("text")
        
        # Store context
        if self.config.store_context and "context_description" in chunk:
            chunk_data["context_description"] = chunk.get("context_description")
            chunk_data["context_generation_method"] = chunk.get("context_generation_method")
            chunk_data["context_token_count"] = chunk.get("context_token_count", 0)
            chunk_data["context_cost"] = chunk.get("context_cost", 0.0)
        
        # Store contextualized text (for embedding)
        if self.config.store_contextualized and "contextualized_text" in chunk:
            chunk_data["contextualized_text"] = chunk.get("contextualized_text")
            chunk_data["text"] = chunk.get("contextualized_text")  # For embedding
        else:
            chunk_data["text"] = chunk.get("text")  # Use original
        
        # Store metadata
        chunk_data["metadata"] = chunk.get("metadata", {})
        
        # Save to database
        self.database.store_chunk(chunk_data)

    def retrieve_chunks(
        self,
        chunk_ids: List[str],
        return_format: str = "original"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with specified format.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            return_format: Format to return ("original", "context", "contextualized", "both")
            
        Returns:
            List of chunks with requested format
        """
        chunks = self.database.get_chunks_by_ids(chunk_ids)
        
        # Format chunks based on return_format
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunk = chunk.copy()
            
            if return_format == "original":
                formatted_chunk["text"] = chunk.get("original_text", chunk.get("text"))
            elif return_format == "contextualized":
                formatted_chunk["text"] = chunk.get("contextualized_text", chunk.get("text"))
            elif return_format == "both":
                formatted_chunk["original_text"] = chunk.get("original_text")
                formatted_chunk["contextualized_text"] = chunk.get("contextualized_text")
                formatted_chunk["context"] = chunk.get("context_description")
            elif return_format == "context":
                formatted_chunk["context"] = chunk.get("context_description")
            
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks
