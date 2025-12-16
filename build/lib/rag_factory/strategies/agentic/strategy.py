"""
Agentic RAG strategy implementation.

This module provides the main strategy class that integrates agent-based
tool selection with the RAG pipeline.
"""

from typing import List, Dict, Any, Optional, Set
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
from ...services.dependencies import StrategyDependencies, ServiceDependency
from ..base import IRAGStrategy
from ...factory import register_rag_strategy as register_strategy

logger = logging.getLogger(__name__)


@register_strategy("AgenticRAGStrategy")
class AgenticRAGStrategy(IRAGStrategy):
    """
    Agentic RAG strategy where an agent selects appropriate tools
    to retrieve information based on query type.
    
    This strategy uses an LLM-powered agent to dynamically choose
    between different retrieval tools (semantic search, document reader,
    metadata search, hybrid search) based on the query characteristics.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """Initialize agentic RAG strategy.
        
        Args:
            config: Strategy configuration dictionary
            dependencies: Injected service dependencies
        """
        # Initialize base class (validates dependencies)
        super().__init__(config, dependencies)
        
        # Note: chunk_repository and document_repository would need to be
        # added to StrategyDependencies or passed via config
        # For now, we'll access services from self.deps
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent
        self.agent = SimpleAgent(self.deps.llm_service, self.tools)
        
        # Initialize query analyzer if enabled
        self.query_analyzer = (
            QueryAnalyzer(self.deps.llm_service)
            if self.config.get('enable_query_analysis', True)
            else None
        )
    
    @property
    def strategy_config(self) -> AgenticStrategyConfig:
        """Get parsed strategy configuration.
        
        Returns:
            Parsed AgenticStrategyConfig object
        """
        return AgenticStrategyConfig(**self.config)
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.
        
        Returns:
            Set of required service dependencies
        """
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}

    def _initialize_tools(self) -> List:
        """Initialize available tools.
        
        Returns:
            List of tool instances
        """
        # Get repositories from database service
        # The database service should have chunk_repository and document_repository attributes
        chunk_repo = getattr(self.deps.database_service, 'chunk_repository', None)
        doc_repo = getattr(self.deps.database_service, 'document_repository', None)
        
        # Create all available tools
        all_tools = []
        
        # Only create tools if we have the required repositories
        if chunk_repo and self.deps.embedding_service:
            all_tools.extend([
                SemanticSearchTool(
                    chunk_repository=chunk_repo,
                    embedding_service=self.deps.embedding_service
                ),
                MetadataSearchTool(
                    chunk_repository=chunk_repo,
                    embedding_service=self.deps.embedding_service
                ),
                HybridSearchTool(
                    chunk_repository=chunk_repo,
                    embedding_service=self.deps.embedding_service
                )
            ])
        
        if doc_repo and chunk_repo:
            all_tools.append(
                DocumentReaderTool(
                    document_repository=doc_repo,
                    chunk_repository=chunk_repo
                )
            )
        
        # Filter by enabled tools if specified
        enabled_tools_config = self.config.get('enabled_tools')
        if enabled_tools_config:
            enabled_names = set(enabled_tools_config)
            all_tools = [t for t in all_tools if t.name in enabled_names]
        
        logger.info(f"Initialized {len(all_tools)} tools: {[t.name for t in all_tools]}")
        return all_tools
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        """Prepare and chunk documents for retrieval.
        
        Args:
            documents: List of documents to prepare
            
        Returns:
            PreparedData container
        """
        # TODO: Implement document preparation for agentic strategy
        raise NotImplementedError("prepare_data not yet implemented for AgenticRAGStrategy")
    
    async def aretrieve(self, query: str, top_k: int):
        """Async retrieve - delegates to sync retrieve for now.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of chunks
        """
        return self.retrieve(query, top_k)
    
    def process_query(self, query: str, context):
        """Process query with context to generate answer.
        
        Args:
            query: User query
            context: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        # TODO: Implement query processing for agentic strategy
        raise NotImplementedError("process_query not yet implemented for AgenticRAGStrategy")

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
            result = self.agent.run(query, max_iterations=self.strategy_config.max_iterations)

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
            if self.strategy_config.fallback_to_semantic:
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
            "config": self.strategy_config.dict()
        }
