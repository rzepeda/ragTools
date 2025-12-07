"""Demo script for self-reflective RAG strategy.

This example demonstrates how to use the self-reflective RAG strategy
to improve retrieval quality through automatic query refinement.
"""

from unittest.mock import Mock
from rag_factory.strategies.self_reflective import SelfReflectiveRAGStrategy
from rag_factory.services.llm import LLMService
from rag_factory.services.llm.config import LLMServiceConfig


def create_mock_base_strategy():
    """Create a mock base retrieval strategy for demonstration."""
    strategy = Mock()
    
    # Simulate poor initial results that improve after refinement
    call_count = 0
    
    def mock_retrieve(query, top_k=5, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First attempt: poor results
            return [
                {"chunk_id": "c1", "text": "Python is a programming language", "score": 0.6},
                {"chunk_id": "c2", "text": "Machine learning uses algorithms", "score": 0.5},
            ]
        else:
            # After refinement: better results
            return [
                {"chunk_id": "c3", "text": "Machine learning is a subset of AI that enables computers to learn from data", "score": 0.95},
                {"chunk_id": "c4", "text": "Deep learning is a type of machine learning using neural networks", "score": 0.9},
                {"chunk_id": "c5", "text": "Supervised learning trains models on labeled data", "score": 0.85},
            ]
    
    strategy.retrieve = mock_retrieve
    return strategy


def main():
    """Run the demo."""
    print("=" * 70)
    print("Self-Reflective RAG Strategy Demo")
    print("=" * 70)
    print()
    
    # Note: This demo uses a mock LLM service
    # To use a real LLM, uncomment the code below and set your API key
    
    # Real LLM setup (requires ANTHROPIC_API_KEY environment variable):
    # llm_config = LLMServiceConfig(
    #     provider="anthropic",
    #     model="claude-3-haiku-20240307"
    # )
    # llm_service = LLMService(llm_config)
    
    # For demo purposes, we'll use a mock LLM
    llm_service = Mock()
    llm_service.complete.side_effect = [
        # First grading call
        Mock(content="""Result 1:
Grade: 2
Relevance: 0.3
Completeness: 0.3
Reasoning: Not very relevant to the query about machine learning

Result 2:
Grade: 3
Relevance: 0.5
Completeness: 0.4
Reasoning: Mentions machine learning but lacks detail
"""),
        # Refinement call
        Mock(content="""Refined Query: What are the key concepts and techniques in machine learning and how does it work?
Strategy: expansion
Reasoning: Adding more specific keywords to get comprehensive results about machine learning
"""),
        # Second grading call
        Mock(content="""Result 1:
Grade: 5
Relevance: 0.95
Completeness: 0.9
Reasoning: Excellent match, comprehensive explanation of machine learning

Result 2:
Grade: 5
Relevance: 0.9
Completeness: 0.85
Reasoning: Very relevant, explains deep learning clearly

Result 3:
Grade: 4
Relevance: 0.85
Completeness: 0.8
Reasoning: Good match, covers supervised learning
""")
    ]
    
    # Create base strategy
    base_strategy = create_mock_base_strategy()
    
    # Create self-reflective strategy
    print("Creating self-reflective RAG strategy...")
    strategy = SelfReflectiveRAGStrategy(
        base_retrieval_strategy=base_strategy,
        llm_service=llm_service,
        config={
            "grade_threshold": 4.0,  # Retry if average grade < 4.0
            "max_retries": 2,
            "timeout_seconds": 10
        }
    )
    print("✓ Strategy created")
    print()
    
    # Perform retrieval
    query = "What is machine learning?"
    print(f"Query: '{query}'")
    print()
    print("Retrieving with self-reflection...")
    print("-" * 70)
    
    results = strategy.retrieve(query, top_k=3)
    
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Text: {result['text']}")
        print(f"  Grade: {result['grade']}/5 ({result['grade_level']})")
        print(f"  Reasoning: {result['grade_reasoning']}")
        print(f"  Combined Score: {result['combined_score']:.3f}")
        print(f"  Retrieved in Attempt: {result['retrieval_attempt']}")
        print()
    
    # Show refinements
    if results and "refinements" in results[0]:
        refinements = results[0]["refinements"]
        if refinements:
            print("=" * 70)
            print("Query Refinements Made")
            print("=" * 70)
            print()
            for ref in refinements:
                print(f"Iteration {ref['iteration']}:")
                print(f"  Refined Query: {ref['refined_query']}")
                print(f"  Strategy: {ref['strategy']}")
                print(f"  Reasoning: {ref['reasoning']}")
                print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total Attempts: {results[0]['total_attempts']}")
    print(f"Final Results: {len(results)}")
    print(f"Average Grade: {sum(r['grade'] for r in results) / len(results):.2f}/5")
    print()
    print("✓ Demo complete!")


if __name__ == "__main__":
    main()
