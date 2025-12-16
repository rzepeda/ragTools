#!/bin/bash
# Test runner for strategy pair tests with timeouts

echo "Testing Strategy Pair Integration Tests"
echo "========================================"
echo ""

TIMEOUT=15
PASSING=0
FAILING=0
TIMEOUT_COUNT=0

# Array of test files
tests=(
    "test_semantic_local_pair.py"
    "test_semantic_api_pair.py"
    "test_fine_tuned_embeddings_pair.py"
    "test_self_reflective_pair.py"
    "test_agentic_rag_pair.py"
    "test_late_chunking_pair.py"
    "test_contextual_retrieval_pair.py"
    "test_context_aware_chunking_pair.py"
    "test_hierarchical_rag_pair.py"
    "test_hybrid_search_pair.py"
    "test_keyword_pair.py"
    "test_knowledge_graph_pair.py"
    "test_multi_query_pair.py"
    "test_query_expansion_pair.py"
    "test_reranking_pair.py"
)

for test in "${tests[@]}"; do
    echo "Testing: $test"
    timeout $TIMEOUT python -m pytest "tests/integration/$test" -v --tb=line -q 2>&1 > /tmp/test_output.txt
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ PASSED"
        ((PASSING++))
    elif [ $exit_code -eq 124 ]; then
        echo "  ⏱️  TIMEOUT"
        ((TIMEOUT_COUNT++))
        ((FAILING++))
    else
        echo "  ❌ FAILED"
        tail -5 /tmp/test_output.txt | sed 's/^/    /'
        ((FAILING++))
    fi
    echo ""
done

echo "========================================"
echo "Summary:"
echo "  Passing: $PASSING/15"
echo "  Failing: $FAILING/15"
echo "  Timeouts: $TIMEOUT_COUNT/15"
echo "========================================"
