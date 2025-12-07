"""
Agentic RAG strategy implementation.

This module provides the main strategy class that integrates agent-based
tool selection with the RAG pipeline.
"""

from typing import List, Dict, Any, Optional
import logging

from .agent import SimpleAgent
from .tool_implementations import (
    SemanticSearchTool,
    DocumentReaderTool,
    MetadataSearchTool,
    HybridSearchTool
)
from .query_analyzer import QueryAnalyzer
from .config import AgenticStrategyConfig
from ...services.llm import LLMService
from ...services.embedding import EmbeddingService
from ...repositories.chunk import ChunkRepository
from ...repositories.document import DocumentRepository

logger = logging.getLogger(__name__)


class AgenticRAGStrategy:
    """
    Agentic RAG strategy where an agent selects appropriate tools
    to retrieve information based on query type.
    
    This strategy uses an LLM-powered agent to dynamically choose
    between different retrieval tools (semantic search, document reader,
    metadata search, hybrid search) based on the query characteristics.
    """

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize agentic RAG strategy.
        
        Args:
            llm_service: LLM service for agent decisions
            embedding_service: Embedding service for vector search
            chunk_repository: Repository for chunk operations
            document_repository: Repository for document operations
            config: Optional configuration dictionary
        """
        # Parse configuration
        if config is None:
            config = {}
        self.config = AgenticStrategyConfig(**config)
        
        # Store services
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository
        self.document_repository = document_repository
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent
        self.agent = SimpleAgent(self.llm_service, self.tools)
        
        # Initialize query analyzer if enabled
        self.query_analyzer = (
            QueryAnalyzer(self.llm_service)
            if self.config.enable_query_analysis
            else None
        )

    def _initialize_tools(self) -> List:
        """Initialize available tools.
        
        Returns:
            List of tool instances
        """
        all_tools = [
            SemanticSearchTool(self.chunk_repository, self.embedding_service),
            DocumentReaderTool(self.document_repository, self.chunk_repository),
            MetadataSearchTool(self.chunk_repository, self.embedding_service),
            HybridSearchTool(self.chunk_repository, self.embedding_service)
        ]
        
        # Filter by enabled tools if specified
        if self.config.enabled_tools:
            enabled_names = set(self.config.enabled_tools)
            all_tools = [t for t in all_tools if t.name in enabled_names]
        
        logger.info(f"Initialized {len(all_tools)} tools: {[t.name for t in all_tools]}")
        return all_tools

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant information using agentic approach.
        
        Args:
            query: User query
            top_k: Maximum results to return
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved chunks/documents with metadata
        """
        logger.info(f"Agentic RAG strategy retrieving for: {query}")

        try:
            # Optional: Analyze query first
            if self.query_analyzer:
                analysis = self.query_analyzer.analyze(query)
                logger.info(
                    f"Query analysis: type={analysis.query_type.value}, "
                    f"complexity={analysis.complexity}, "
                    f"recommended_tools={analysis.recommended_tools}"
                )

            # Run agent
            result = self.agent.run(query, max_iterations=self.config.max_iterations)

            # Extract results
            chunks = result["results"][:top_k]

            # Add strategy metadata
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk["strategy"] = "agentic"
                    chunk["agent_trace"] = result["trace"]

            logger.info(
                f"Agentic retrieval completed: {len(chunks)} results, "
                f"{result['trace']['iterations']} iterations, "
                f"{len(result['trace']['tool_calls'])} tool calls"
            )

            return chunks

        except Exception as e:
            logger.error(f"Agentic retrieval failed: {e}", exc_info=True)

            # Fallback to semantic search if enabled
            if self.config.fallback_to_semantic:
                logger.info("Falling back to semantic search")
                return self._fallback_search(query, top_k)

            raise

    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to simple semantic search.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of chunks
        """
        try:
            # Use semantic search tool directly
            semantic_tool = next(
                (t for t in self.tools if t.name == "semantic_search"),
                None
            )
            
            if not semantic_tool:
                logger.error("Semantic search tool not available for fallback")
                return []
            
            result = semantic_tool.execute(query=query, top_k=top_k)

            if result.success:
                # Add strategy metadata
                for chunk in result.data:
                    if isinstance(chunk, dict):
                        chunk["strategy"] = "agentic_fallback"
                return result.data
            else:
                logger.error(f"Fallback search failed: {result.error}")
                return []
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []

    @property
    def name(self) -> str:
        """Strategy name.
        
        Returns:
            Strategy identifier
        """
        return "agentic"

    @property
    def description(self) -> str:
        """Strategy description.
        
        Returns:
            Human-readable description
        """
        return "Agent-based retrieval with dynamic tool selection"

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics.
        
        Returns:
            Dictionary with strategy stats
        """
        return {
            "name": self.name,
            "description": self.description,
            "num_tools": len(self.tools),
            "tools": [t.name for t in self.tools],
            "config": self.config.dict()
        }
