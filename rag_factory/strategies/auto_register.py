"""Import all strategies to ensure they are registered."""

# Import all indexing strategies to trigger registration
from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
from rag_factory.strategies.indexing.hierarchical import HierarchicalIndexing
from rag_factory.strategies.indexing.context_aware import ContextAwareChunkingIndexing
from rag_factory.strategies.indexing.keyword_indexing import KeywordIndexing
from rag_factory.strategies.indexing.knowledge_graph_indexing import KnowledgeGraphIndexing

# Import basic retrieval strategies (these exist)
try:
    from rag_factory.strategies.retrieval.semantic_retriever import SemanticRetriever
except ImportError as e:
    print(f"Could not import SemanticRetriever: {e}")

try:
    from rag_factory.strategies.retrieval.keyword_retriever import KeywordRetriever
except ImportError as e:
    print(f"Could not import KeywordRetriever: {e}")

try:
    from rag_factory.strategies.retrieval.knowledge_graph_retriever import KnowledgeGraphRetriever
except ImportError as e:
    print(f"Could not import KnowledgeGraphRetriever: {e}")

try:
    from rag_factory.strategies.retrieval.multi_query_retriever import MultiQueryRetriever
except ImportError as e:
    print(f"Could not import MultiQueryRetriever: {e}")

try:
    from rag_factory.strategies.retrieval.query_expansion_retriever import QueryExpansionRetriever
except ImportError as e:
    print(f"Could not import QueryExpansionRetriever: {e}")

try:
    from rag_factory.strategies.retrieval.hybrid_retriever import HybridSearchRetriever
    print("✅ HybridSearchRetriever imported")
except ImportError as e:
    print(f"❌ Could not import HybridSearchRetriever: {e}")

# Import complex strategies (all exist!)
try:
    from rag_factory.strategies.agentic.strategy import AgenticRAGStrategy
    print("✅ AgenticRAGStrategy imported")
except ImportError as e:
    print(f"❌ Could not import AgenticRAGStrategy: {e}")

try:
    from rag_factory.strategies.late_chunking.strategy import LateChunkingRAGStrategy
    print("✅ LateChunkingRAGStrategy imported")
except ImportError as e:
    print(f"❌ Could not import LateChunkingRAGStrategy: {e}")

try:
    from rag_factory.strategies.self_reflective.strategy import SelfReflectiveRAGStrategy
    print("✅ SelfReflectiveRAGStrategy imported")
except ImportError as e:
    print(f"❌ Could not import SelfReflectiveRAGStrategy: {e}")

# Import other complex strategies (may not exist yet)
try:
    from rag_factory.strategies.retrieval.hierarchical import HierarchicalRetriever
except ImportError:
    pass

try:
    from rag_factory.strategies.retrieval.hybrid import HybridRetriever
except ImportError:
    pass

try:
    from rag_factory.strategies.reranking.strategy import RerankingRetriever
except ImportError:
    pass

try:
    from rag_factory.strategies.contextual.strategy import ContextualRetrievalStrategy
except ImportError:
    pass

try:
    from rag_factory.strategies.fine_tuned.strategy import FineTunedEmbeddingStrategy
except ImportError:
    pass

__all__ = [
    'VectorEmbeddingIndexing',
    'HierarchicalIndexing',
    'ContextAwareChunkingIndexing',
    'KeywordIndexing',
    'KnowledgeGraphIndexing',
    'SemanticRetriever',
    'KeywordRetriever',
    'AgenticRAGStrategy',
    'LateChunkingRAGStrategy',
    'SelfReflectiveRAGStrategy',
]
