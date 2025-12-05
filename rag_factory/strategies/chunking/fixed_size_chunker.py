"""Fixed-size chunking strategy (baseline implementation)."""

from typing import List, Dict, Optional
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod

logger = logging.getLogger(__name__)


class FixedSizeChunker(IChunker):
    """Simple fixed-size chunking strategy.

    Splits documents into chunks of approximately equal size with optional overlap.
    This is the baseline approach - fast but doesn't consider semantics or structure.

    Attributes:
        config: Chunking configuration
        tokenizer: Tokenizer for counting tokens
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize fixed-size chunker.

        Args:
            config: Chunking configuration
        """
        super().__init__(config)

        if tiktoken is None:
            logger.warning(
                "tiktoken not installed. Using word count approximation. "
                "Install with: pip install tiktoken"
            )
            self.tokenizer = None
        else:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback.")
                self.tokenizer = None

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document into fixed-size pieces.

        Args:
            document: Document text to chunk
            document_id: Unique document identifier

        Returns:
            List of Chunk objects
        """
        if not document or not document.strip():
            return []

        # Split into words for simple chunking
        words = document.split()
        chunks = []
        current_words = []
        current_tokens = 0
        chunk_idx = 0

        for word in words:
            word_tokens = self._count_tokens(word)

            # Check if adding this word would exceed target size
            if current_tokens + word_tokens > self.config.target_chunk_size and current_words:
                # Create chunk from accumulated words
                chunk_text = " ".join(current_words)
                chunk = self._create_chunk(
                    chunk_text,
                    document_id,
                    chunk_idx,
                    current_tokens
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Handle overlap
                if self.config.chunk_overlap > 0:
                    # Keep last N words for overlap
                    overlap_words = self._get_overlap_words(
                        current_words,
                        self.config.chunk_overlap
                    )
                    current_words = overlap_words + [word]
                    current_tokens = self._count_tokens(" ".join(current_words))
                else:
                    current_words = [word]
                    current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens

        # Add remaining words as final chunk
        if current_words:
            chunk_text = " ".join(current_words)
            chunk = self._create_chunk(
                chunk_text,
                document_id,
                chunk_idx,
                current_tokens
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents.

        Args:
            documents: List of dicts with 'text' and 'id' keys

        Returns:
            List of chunk lists, one per document
        """
        return [
            self.chunk_document(doc["text"], doc["id"])
            for doc in documents
        ]

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        position: int,
        token_count: Optional[int] = None
    ) -> Chunk:
        """Create a chunk with metadata.

        Args:
            text: Chunk text
            document_id: Document identifier
            position: Chunk position
            token_count: Optional pre-calculated token count

        Returns:
            Chunk object
        """
        if token_count is None:
            token_count = self._count_tokens(text)

        metadata = ChunkMetadata(
            chunk_id=f"{document_id}_chunk_{position}",
            source_document_id=document_id,
            position=position,
            start_char=0,
            end_char=len(text),
            section_hierarchy=[],
            chunking_method=ChunkingMethod.FIXED_SIZE,
            token_count=token_count
        )

        return Chunk(text=text, metadata=metadata)

    def _get_overlap_words(self, words: List[str], overlap_tokens: int) -> List[str]:
        """Get last N words that approximately equal overlap_tokens.

        Args:
            words: List of words
            overlap_tokens: Target overlap in tokens

        Returns:
            List of overlap words
        """
        if not words or overlap_tokens <= 0:
            return []

        overlap_words = []
        tokens = 0

        # Start from end and work backwards
        for word in reversed(words):
            word_tokens = self._count_tokens(word)
            if tokens + word_tokens > overlap_tokens:
                break
            overlap_words.insert(0, word)
            tokens += word_tokens

        return overlap_words

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to simple word count approximation (avg 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)
