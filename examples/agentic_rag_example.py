"""
Example usage of the Agentic RAG Strategy.

This example demonstrates how to use the agentic RAG strategy
to perform intelligent retrieval with dynamic tool selection.
"""

from rag_factory.strategies.agentic import AgenticRAGStrategy
from rag_factory.services.llm import LLMService, LLMServiceConfig
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig
from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.repositories.document import DocumentRepository
from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.config import DatabaseConfig


def main():
    """Example of using agentic RAG strategy."""
    
    # 1. Setup database connection
    db_config = DatabaseConfig(
        database_url="postgresql://user:pass@localhost/ragdb"
    )
    db = DatabaseConnection(db_config)
    
    # 2. Initialize repositories
    with db.get_session() as session:
        chunk_repo = ChunkRepository(session)
        doc_repo = DocumentRepository(session)
        
        # 3. Setup LLM service for agent
        llm_config = LLMServiceConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            provider_config={"api_key": "your-api-key"}
        )
        llm_service = LLMService(llm_config)
        
        # 4. Setup embedding service for vector search
        embedding_config = EmbeddingServiceConfig(
            provider="openai",
            model="text-embedding-3-small",
            provider_config={"api_key": "your-api-key"}
        )
        embedding_service = EmbeddingService(embedding_config)
        
        # 5. Create agentic strategy
        strategy = AgenticRAGStrategy(
            llm_service=llm_service,
            embedding_service=embedding_service,
            chunk_repository=chunk_repo,
            document_repository=doc_repo,
            config={
                "max_iterations": 3,
                "enable_query_analysis": True,
                "fallback_to_semantic": True
            }
        )
        
        # 6. Perform retrieval
        query = "What are the authentication best practices?"
        results = strategy.retrieve(query, top_k=5)
        
        # 7. Display results
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Score: {result.get('score', 'N/A')}")
            print(f"  Chunk ID: {result['chunk_id']}")
            print()
        
        # 8. View agent trace
        if results and "agent_trace" in results[0]:
            trace = results[0]["agent_trace"]
            print("\nAgent Execution Trace:")
            print(f"  Iterations: {trace['iterations']}")
            print(f"  Tool Calls: {len(trace['tool_calls'])}")
            
            for tool_call in trace['tool_calls']:
                print(f"    - {tool_call['tool']} (iteration {tool_call['iteration']})")
            
            print(f"\n  Plan: {trace['plan'].get('reasoning', 'N/A')}")


def example_with_different_query_types():
    """Examples of different query types and how the agent handles them."""
    
    # Setup (same as above, abbreviated)
    # ... setup code ...
    
    queries = [
        # Factual query - will use semantic search
        "What is machine learning?",
        
        # Metadata query - will use metadata search
        "Find documents from 2024 by John Doe",
        
        # Specific document query - will use document reader
        "Show me document ID abc-123-def",
        
        # Complex query - may use multiple tools
        "How does the authentication API work and what are the security considerations?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # results = strategy.retrieve(query, top_k=5)
        # ... process results ...


if __name__ == "__main__":
    main()
