"""
Concrete tool implementations for agentic RAG.

This module provides specific tools that agents can use:
- SemanticSearchTool: Search by semantic similarity
- DocumentReaderTool: Read full documents by ID
- MetadataSearchTool: Search by metadata filters
- HybridSearchTool: Combine semantic and metadata search
"""

from typing import List, Dict, Any, Optional
import logging
from uuid import UUID

from .tools import Tool, ToolParameter, ToolResult
from ...repositories.chunk import ChunkRepository
from ...repositories.document import DocumentRepository
from ...services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticSearchTool(Tool):
    """Tool for semantic similarity search.
    
    Uses vector embeddings to find semantically similar chunks.
    """

    def __init__(self, chunk_repository: ChunkRepository, embedding_service: EmbeddingService):
        """Initialize semantic search tool.
        
        Args:
            chunk_repository: Repository for chunk operations
            embedding_service: Service for generating embeddings
        """
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "semantic_search"

    @property
    def description(self) -> str:
        return """Search for relevant text chunks using semantic similarity.
        Use this when you need to find information based on meaning and context,
        not just exact keyword matches. Good for answering questions like
        'What is...' or 'How does... work?' or 'Explain...'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query to find relevant chunks",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5
            ),
            ToolParameter(
                name="min_score",
                type="number",
                description="Minimum similarity score 0-1 (default: 0.7)",
                required=False,
                default=0.7
            )
        ]

    def execute(self, query: str, top_k: int = 5, min_score: float = 0.7) -> ToolResult:
        """Execute semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity threshold
            
        Returns:
            ToolResult with matching chunks
        """
        import time
        start = time.time()

        try:
            # Generate embedding for query
            embedding = self.embedding_service.embed_text(query)
            
            # Search for similar chunks
            results = self.chunk_repository.search_similar(
                embedding=embedding,
                top_k=top_k,
                threshold=min_score
            )

            # Convert to dict format
            chunks = []
            for chunk, score in results:
                chunks.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "text": chunk.text,
                    "score": score,
                    "metadata": chunk.metadata_,
                    "chunk_index": chunk.chunk_index
                })

            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=chunks,
                execution_time=execution_time,
                metadata={
                    "num_results": len(chunks),
                    "query": query,
                    "top_k": top_k,
                    "min_score": min_score
                }
            )
        except Exception as e:
            execution_time = time.time() - start
            logger.error(f"Semantic search failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )


class DocumentReaderTool(Tool):
    """Tool for reading full documents by ID or title."""

    def __init__(self, document_repository: DocumentRepository, chunk_repository: ChunkRepository):
        """Initialize document reader tool.
        
        Args:
            document_repository: Repository for document operations
            chunk_repository: Repository for chunk operations
        """
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository

    @property
    def name(self) -> str:
        return "read_document"

    @property
    def description(self) -> str:
        return """Read the full content of a specific document by ID.
        Use this when you need to read an entire document or when the user
        asks about a specific named document. Example: 'Show me document ID abc123'
        or 'What's in the user guide?'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="document_id",
                type="string",
                description="Document UUID to read",
                required=True
            )
        ]

    def execute(self, document_id: str) -> ToolResult:
        """Read document by ID.
        
        Args:
            document_id: UUID of document to read
            
        Returns:
            ToolResult with document and its chunks
        """
        import time
        start = time.time()

        try:
            # Get document
            doc_uuid = UUID(document_id)
            document = self.document_repository.get_by_id(doc_uuid)
            
            if not document:
                execution_time = time.time() - start
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data=None,
                    error=f"Document {document_id} not found",
                    execution_time=execution_time
                )

            # Get all chunks for this document
            chunks = self.chunk_repository.get_by_document(doc_uuid)

            # Build result
            result = {
                "document_id": str(document.document_id),
                "filename": document.filename,
                "source_path": document.source_path,
                "metadata": document.metadata_,
                "total_chunks": document.total_chunks,
                "status": document.status,
                "chunks": [
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "metadata": chunk.metadata_
                    }
                    for chunk in chunks
                ]
            }

            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={
                    "document_id": document_id,
                    "num_chunks": len(chunks)
                }
            )
        except ValueError as e:
            execution_time = time.time() - start
            logger.error(f"Invalid document ID: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error=f"Invalid document ID format: {document_id}",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start
            logger.error(f"Document read failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )


class MetadataSearchTool(Tool):
    """Tool for searching by metadata filters."""

    def __init__(self, chunk_repository: ChunkRepository, embedding_service: EmbeddingService):
        """Initialize metadata search tool.
        
        Args:
            chunk_repository: Repository for chunk operations
            embedding_service: Service for generating embeddings
        """
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "metadata_search"

    @property
    def description(self) -> str:
        return """Search documents by metadata like author, date, category, tags.
        Use this when the query asks for documents with specific properties.
        Example: 'Find documents from 2024' or 'Show papers by John Doe'
        or 'Get documents tagged with Python'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query for semantic matching",
                required=True
            ),
            ToolParameter(
                name="metadata_filter",
                type="object",
                description="Metadata filters as key-value pairs (e.g., {'author': 'John', 'year': '2024'})",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5
            ),
            ToolParameter(
                name="min_score",
                type="number",
                description="Minimum similarity score 0-1 (default: 0.5)",
                required=False,
                default=0.5
            )
        ]

    def execute(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        top_k: int = 5,
        min_score: float = 0.5
    ) -> ToolResult:
        """Execute metadata search.
        
        Args:
            query: Search query
            metadata_filter: Metadata filters
            top_k: Number of results
            min_score: Minimum similarity threshold
            
        Returns:
            ToolResult with matching chunks
        """
        import time
        start = time.time()

        try:
            # Generate embedding for query
            embedding = self.embedding_service.embed_text(query)
            
            # Search with metadata filter
            results = self.chunk_repository.search_similar_with_metadata(
                embedding=embedding,
                top_k=top_k,
                metadata_filter=metadata_filter,
                threshold=min_score
            )

            # Convert to dict format
            chunks = []
            for chunk, score in results:
                chunks.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "text": chunk.text,
                    "score": score,
                    "metadata": chunk.metadata_,
                    "chunk_index": chunk.chunk_index
                })

            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=chunks,
                execution_time=execution_time,
                metadata={
                    "num_results": len(chunks),
                    "query": query,
                    "metadata_filter": metadata_filter,
                    "top_k": top_k
                }
            )
        except Exception as e:
            execution_time = time.time() - start
            logger.error(f"Metadata search failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )


class HybridSearchTool(Tool):
    """Tool for hybrid search combining semantic and keyword matching."""

    def __init__(self, chunk_repository: ChunkRepository, embedding_service: EmbeddingService):
        """Initialize hybrid search tool.
        
        Args:
            chunk_repository: Repository for chunk operations
            embedding_service: Service for generating embeddings
        """
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "hybrid_search"

    @property
    def description(self) -> str:
        return """Combine semantic similarity search with keyword matching.
        Use this when you need both conceptual understanding and exact term matching.
        Good for technical queries that require specific terminology."""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5
            ),
            ToolParameter(
                name="semantic_weight",
                type="number",
                description="Weight for semantic score 0-1 (default: 0.7)",
                required=False,
                default=0.7
            )
        ]

    def execute(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> ToolResult:
        """Execute hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic vs keyword (0-1)
            
        Returns:
            ToolResult with matching chunks
        """
        import time
        start = time.time()

        try:
            # Generate embedding for query
            embedding = self.embedding_service.embed_text(query)
            
            # Get semantic results (more than needed for reranking)
            semantic_results = self.chunk_repository.search_similar(
                embedding=embedding,
                top_k=top_k * 2,
                threshold=0.0
            )

            # Simple keyword matching (case-insensitive)
            query_terms = set(query.lower().split())
            
            # Combine scores
            combined_results = []
            for chunk, semantic_score in semantic_results:
                # Calculate keyword score
                chunk_terms = set(chunk.text.lower().split())
                keyword_score = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
                
                # Weighted combination
                hybrid_score = (
                    semantic_weight * semantic_score +
                    (1 - semantic_weight) * keyword_score
                )
                
                combined_results.append((chunk, hybrid_score))

            # Sort by hybrid score and take top_k
            combined_results.sort(key=lambda x: x[1], reverse=True)
            combined_results = combined_results[:top_k]

            # Convert to dict format
            chunks = []
            for chunk, score in combined_results:
                chunks.append({
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "text": chunk.text,
                    "score": score,
                    "metadata": chunk.metadata_,
                    "chunk_index": chunk.chunk_index
                })

            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=chunks,
                execution_time=execution_time,
                metadata={
                    "num_results": len(chunks),
                    "query": query,
                    "top_k": top_k,
                    "semantic_weight": semantic_weight
                }
            )
        except Exception as e:
            execution_time = time.time() - start
            logger.error(f"Hybrid search failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )
