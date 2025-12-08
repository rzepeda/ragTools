from typing import List, Set, Optional, Dict, Any
from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.services.dependencies import ServiceDependency

class IndexingPipeline:
    """Pipeline for executing indexing strategies"""
    
    def __init__(
        self,
        strategies: List[IIndexingStrategy],
        context: IndexingContext
    ):
        """
        Create indexing pipeline.
        
        Args:
            strategies: Ordered list of indexing strategies
            context: Shared indexing context
        """
        self.strategies = strategies
        self.context = context
        self._last_result: Optional[IndexingResult] = None
    
    def get_capabilities(self) -> Set[IndexCapability]:
        """
        Get combined capabilities from all strategies.
        
        Returns:
            Union of all strategy capabilities
        """
        if self._last_result:
            return self._last_result.capabilities
        
        # Return declared capabilities if not executed yet
        all_caps = set()
        for strategy in self.strategies:
            all_caps.update(strategy.produces())
        return all_caps
    
    async def index(
        self,
        documents: List[Dict[str, Any]]
    ) -> IndexingResult:
        """
        Execute indexing pipeline.
        
        Args:
            documents: Documents to index
            
        Returns:
            Combined IndexingResult from all strategies
        """
        all_capabilities = set()
        all_metadata = {}
        total_chunks = 0
        
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            
            # Execute strategy
            result = await strategy.process(documents, self.context)
            
            # Aggregate results
            all_capabilities.update(result.capabilities)
            all_metadata[strategy_name] = result.metadata
            total_chunks = max(total_chunks, result.chunk_count)
        
        self._last_result = IndexingResult(
            capabilities=all_capabilities,
            metadata=all_metadata,
            document_count=len(documents),
            chunk_count=total_chunks
        )
        
        return self._last_result


class RetrievalPipeline:
    """Pipeline for executing retrieval strategies"""
    
    def __init__(
        self,
        strategies: List[IRetrievalStrategy],
        context: RetrievalContext
    ):
        """
        Create retrieval pipeline.
        
        Args:
            strategies: Ordered list of retrieval strategies
            context: Shared retrieval context
        """
        self.strategies = strategies
        self.context = context
    
    def get_requirements(self) -> Set[IndexCapability]:
        """
        Get combined requirements from all strategies.
        
        Returns:
            Union of all strategy requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires())
        return all_reqs
    
    def get_service_requirements(self) -> Set[ServiceDependency]:
        """
        Get combined service requirements from all strategies.
        
        Returns:
            Union of all service requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires_services())
        return all_reqs
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Any]: # Using Any for Chunk to avoid circular imports if not imported
        """
        Execute retrieval pipeline.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Retrieved chunks after all strategies applied
        """
        current_query = query
        results = []
        
        for strategy in self.strategies:
            # Each strategy processes the query
            # In this implementation, we execute them in sequence
            # The last strategy's result is returned
            results = await strategy.retrieve(current_query, self.context, top_k)
        
        return results
