# Epic 7 Verification Guide

This document provides comprehensive verification procedures for all Epic 7 strategies: Knowledge Graph, Late Chunking, and Fine-Tuned Embeddings.

---

## Table of Contents

1. [Overview](#overview)
2. [Story 7.1: Knowledge Graph Verification](#story-71-knowledge-graph-verification)
3. [Story 7.2: Late Chunking Verification](#story-72-late-chunking-verification)
4. [Story 7.3: Fine-Tuned Embeddings Verification](#story-73-fine-tuned-embeddings-verification)
5. [Epic-Level Integration Tests](#epic-level-integration-tests)
6. [Performance Validation](#performance-validation)
7. [Quality Metrics](#quality-metrics)

---

## Overview

### Verification Levels

1. **Unit Tests** - Individual component testing (>85% coverage required)
2. **Integration Tests** - End-to-end workflow testing
3. **Performance Tests** - Latency and throughput validation
4. **Quality Tests** - Accuracy and effectiveness metrics
5. **Comparison Tests** - Baseline vs. experimental strategies

### Success Criteria Summary

| Strategy | Key Metrics | Target |
|----------|-------------|--------|
| Knowledge Graph | Entity F1, Retrieval Improvement | >0.80, >10% |
| Late Chunking | Coherence Score, Context Preservation | >0.85, >Traditional |
| Fine-Tuned Embeddings | Accuracy Improvement, A/B Test Pass | >5%, Statistical Significance |

---

## Story 7.1: Knowledge Graph Verification

### Pre-requisites

```bash
# Install dependencies
pip install neo4j networkx pyvis spacy
python -m spacy download en_core_web_sm

# Verify installation
python -c "import neo4j; import networkx; print('Dependencies OK')"
```

### Unit Test Verification

```bash
# Run knowledge graph unit tests
pytest tests/unit/strategies/knowledge_graph/ -v --cov=rag_factory/strategies/knowledge_graph

# Expected output:
# - test_entity_extraction: PASSED
# - test_relationship_extraction: PASSED
# - test_graph_store_operations: PASSED
# - test_hybrid_retrieval: PASSED
# - Coverage: >85%
```

### Integration Test Verification

#### Test 1: Entity Extraction

```python
# tests/integration/test_kg_entity_extraction.py
from rag_factory.strategies.knowledge_graph import EntityExtractor

def test_entity_extraction_quality():
    """Verify entity extraction quality."""
    extractor = EntityExtractor(llm_service, config)

    test_doc = """
    Python is a programming language created by Guido van Rossum.
    It is widely used at Google and Facebook for machine learning.
    """

    entities = extractor.extract_entities(test_doc, "test_doc")

    # Verify entities extracted
    entity_names = [e.name for e in entities]
    assert "Python" in entity_names
    assert "Guido van Rossum" in entity_names
    assert "Google" in entity_names

    # Verify entity types
    python_entity = next(e for e in entities if e.name == "Python")
    assert python_entity.type in [EntityType.CONCEPT, EntityType.OBJECT]

    # Verify confidence scores
    assert all(0.0 <= e.confidence <= 1.0 for e in entities)

    print("✓ Entity extraction quality verified")
```

**Expected Results:**
- Extract 4-6 entities from test document
- Correct entity types assigned
- Confidence scores in valid range (0.0-1.0)
- Entity F1 score >0.80 on test set

#### Test 2: Graph Construction and Traversal

```python
def test_graph_construction_and_traversal():
    """Verify graph building and traversal."""
    from rag_factory.strategies.knowledge_graph import KnowledgeGraphRAGStrategy

    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=vector_store,
        llm_service=llm,
        config={"graph_backend": "memory"}
    )

    # Index documents
    docs = [
        ("Python is used for machine learning.", "doc1"),
        ("Machine learning is a subset of AI.", "doc2"),
        ("AI enables intelligent systems.", "doc3")
    ]

    for text, doc_id in docs:
        strategy.index_document(text, doc_id)

    # Verify graph stats
    stats = strategy.graph_store.get_stats()
    assert stats["num_entities"] >= 3  # At least Python, ML, AI
    assert stats["num_relationships"] >= 2  # Connections

    # Test graph traversal
    results = strategy.retrieve("What is Python used for?", top_k=3)

    # Verify results include graph information
    assert len(results) > 0
    assert any("related_entities" in r for r in results)
    assert any("relationship_paths" in r for r in results)

    print("✓ Graph construction and traversal verified")
```

**Expected Results:**
- Graph contains extracted entities
- Relationships correctly linked
- Traversal returns connected entities
- Hybrid scores combine vector + graph

#### Test 3: Hybrid Search Performance

```python
def test_hybrid_search_performance():
    """Verify hybrid search improves over vector-only."""
    # Setup two strategies: vector-only and hybrid
    vector_only = VectorSearchStrategy(vector_store)
    hybrid_kg = KnowledgeGraphRAGStrategy(vector_store, llm, config)

    # Index same documents in both
    test_corpus = load_test_corpus()  # 100 documents with entities

    for doc in test_corpus:
        vector_only.index_document(doc["text"], doc["id"])
        hybrid_kg.index_document(doc["text"], doc["id"])

    # Run benchmark queries
    test_queries = [
        "What causes climate change?",
        "How is Python related to machine learning?",
        "Who created Linux?"
    ]

    vector_scores = []
    hybrid_scores = []

    for query in test_queries:
        # Ground truth from annotations
        ground_truth = test_corpus_annotations[query]

        # Vector-only results
        v_results = vector_only.retrieve(query, top_k=5)
        v_score = calculate_relevance(v_results, ground_truth)
        vector_scores.append(v_score)

        # Hybrid results
        h_results = hybrid_kg.retrieve(query, top_k=5)
        h_score = calculate_relevance(h_results, ground_truth)
        hybrid_scores.append(h_score)

    # Verify improvement
    avg_vector = np.mean(vector_scores)
    avg_hybrid = np.mean(hybrid_scores)

    improvement = ((avg_hybrid - avg_vector) / avg_vector) * 100

    assert improvement > 10, f"Hybrid improvement: {improvement:.1f}% (expected >10%)"

    print(f"✓ Hybrid search verified: {improvement:.1f}% improvement")
```

**Expected Results:**
- Hybrid search shows >10% improvement over vector-only
- Relationship queries perform better with graph
- Latency <500ms for hybrid search

### Performance Benchmarks

```bash
# Run performance benchmarks
pytest tests/benchmarks/test_kg_performance.py -v

# Check outputs:
# - Entity extraction: <2s per document
# - Graph insertion: <100ms per entity
# - Hybrid search: <500ms end-to-end
# - Graph traversal (3-hop): <200ms
```

### Manual Verification Checklist

- [ ] Entity extraction produces reasonable entities for test documents
- [ ] Relationships make semantic sense
- [ ] Graph visualization shows expected structure (use pyvis)
- [ ] Hybrid search returns relevant results
- [ ] Performance meets requirements
- [ ] Error handling works (malformed documents, LLM failures)
- [ ] Documentation complete with examples

---

## Story 7.2: Late Chunking Verification

### Pre-requisites

```bash
# Install dependencies
pip install transformers torch tokenizers sentence-transformers

# Verify long-context model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2'); print('Model OK')"
```

### Unit Test Verification

```bash
# Run late chunking unit tests
pytest tests/unit/strategies/late_chunking/ -v --cov=rag_factory/strategies/late_chunking

# Expected output:
# - test_document_embedder: PASSED
# - test_token_embedding_extraction: PASSED
# - test_embedding_chunker: PASSED
# - test_text_reconstruction: PASSED
# - Coverage: >85%
```

### Integration Test Verification

#### Test 1: End-to-End Late Chunking

```python
def test_late_chunking_workflow():
    """Verify complete late chunking workflow."""
    from rag_factory.strategies.late_chunking import LateChunkingRAGStrategy

    config = {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "chunking_method": "semantic_boundary",
        "target_chunk_size": 256,
        "compute_coherence_scores": True
    }

    strategy = LateChunkingRAGStrategy(vector_store, config)

    # Test document
    document = """
    Machine learning is a field of artificial intelligence that uses statistical
    techniques to give computer systems the ability to learn from data.

    Deep learning is a subset of machine learning that uses neural networks with
    multiple layers. These networks can learn hierarchical representations of data.

    Applications of deep learning include image recognition, natural language
    processing, and speech recognition. The field has seen rapid advancement.
    """

    # Index with late chunking
    strategy.index_document(document, "ml_doc")

    # Retrieve
    results = strategy.retrieve("What is deep learning?", top_k=3)

    # Verify results
    assert len(results) > 0
    assert all("strategy" in r for r in results)
    assert all(r["strategy"] == "late_chunking" for r in results)

    # Verify metadata
    for result in results:
        metadata = result.get("metadata", {})
        assert "coherence_score" in metadata
        assert "token_range" in metadata
        assert "chunking_method" in metadata

    print("✓ Late chunking workflow verified")
```

**Expected Results:**
- Document embedded successfully
- Token embeddings extracted
- Chunks created from embeddings
- Text reconstruction accurate
- Coherence scores calculated

#### Test 2: Context Preservation

```python
def test_context_preservation():
    """Verify late chunking preserves context better than traditional."""
    # Document designed to test context preservation
    document = """
    The patient presented with acute symptoms. The physician noted elevated markers.
    Laboratory tests confirmed the initial diagnosis. Treatment was initiated immediately.
    The patient's condition improved rapidly over the next 48 hours.
    """

    # Traditional chunking (baseline)
    traditional_strategy = SemanticChunker(traditional_config, embedding_service)
    traditional_chunks = traditional_strategy.chunk_document(document, "patient_doc")

    # Late chunking
    late_strategy = LateChunkingRAGStrategy(vector_store, late_config)
    late_strategy.index_document(document, "patient_doc")

    # Query requiring context
    query = "What happened after the diagnosis?"

    # Compare results
    traditional_results = traditional_vector_search(query, traditional_chunks)
    late_results = late_strategy.retrieve(query, top_k=3)

    # Verify late chunking provides better context
    # (Would need human evaluation or automated relevance scoring)
    late_text = " ".join([r["text"] for r in late_results])

    # Late chunking should include more contextual information
    assert "diagnosis" in late_text.lower()
    assert "treatment" in late_text.lower() or "improved" in late_text.lower()

    print("✓ Context preservation verified")
```

**Expected Results:**
- Late chunking maintains document-level context
- Chunks don't break mid-sentence or mid-thought
- Coherence scores >0.85
- Context information preserved across chunk boundaries

#### Test 3: Text Reconstruction Accuracy

```python
def test_text_reconstruction():
    """Verify text reconstruction is accurate."""
    embedder = DocumentEmbedder(config)
    chunker = EmbeddingChunker(config)

    original_text = "This is a test document with multiple sentences. Each sentence is important."

    # Embed and chunk
    doc_emb = embedder.embed_document(original_text, "test_doc")
    chunks = chunker.chunk_embeddings(doc_emb)

    # Reconstruct text from chunks
    reconstructed_text = " ".join([chunk.text for chunk in chunks])

    # Verify accuracy (allowing for minor whitespace differences)
    assert original_text.replace(" ", "") == reconstructed_text.replace(" ", "")

    # Verify no character loss
    for chunk in chunks:
        start, end = chunk.char_range
        assert original_text[start:end] == chunk.text

    print("✓ Text reconstruction verified")
```

**Expected Results:**
- No character loss during reconstruction
- Text matches original exactly (except whitespace)
- Chunk boundaries align with text boundaries

### Performance Benchmarks

```bash
# Run performance benchmarks
pytest tests/benchmarks/test_late_chunking_performance.py -v

# Check outputs:
# - Document embedding: <2s for 2K tokens
# - Token extraction: <500ms
# - Embedding chunking: <300ms
# - End-to-end: <3s total
```

### Comparison with Traditional Chunking

```python
def test_late_vs_traditional_comparison():
    """Compare late chunking with traditional chunking."""
    test_suite = load_benchmark_suite()  # 50 test documents

    metrics = {
        "traditional": {"coherence": [], "retrieval_acc": []},
        "late": {"coherence": [], "retrieval_acc": []}
    }

    for doc in test_suite:
        # Traditional
        trad_chunks = traditional_chunker.chunk(doc["text"])
        trad_coherence = calculate_coherence(trad_chunks)
        metrics["traditional"]["coherence"].append(trad_coherence)

        # Late
        late_chunks = late_chunker.chunk(doc["text"])
        late_coherence = calculate_coherence(late_chunks)
        metrics["late"]["coherence"].append(late_coherence)

    # Compare averages
    trad_avg = np.mean(metrics["traditional"]["coherence"])
    late_avg = np.mean(metrics["late"]["coherence"])

    improvement = ((late_avg - trad_avg) / trad_avg) * 100

    print(f"Coherence improvement: {improvement:.1f}%")
    assert improvement > 5, "Late chunking should show coherence improvement"

    print("✓ Comparison verified")
```

**Expected Results:**
- Late chunking shows >5% coherence improvement
- Quality justifies additional complexity
- Performance acceptable for use cases

### Manual Verification Checklist

- [ ] Full document embedding works
- [ ] Token embeddings extracted correctly
- [ ] Chunking strategies produce reasonable chunks
- [ ] Text reconstruction accurate
- [ ] Coherence scores calculated
- [ ] Performance meets requirements
- [ ] Comparison with traditional chunking documented
- [ ] Use cases and limitations documented

---

## Story 7.3: Fine-Tuned Embeddings Verification

### Pre-requisites

```bash
# Install dependencies
pip install sentence-transformers transformers torch mlflow scipy

# Verify model loading
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('Model OK')"
```

### Unit Test Verification

```bash
# Run fine-tuned embeddings unit tests
pytest tests/unit/models/embedding/ -v --cov=rag_factory/models/embedding
pytest tests/unit/models/evaluation/ -v --cov=rag_factory/models/evaluation

# Expected output:
# - test_model_registry: PASSED
# - test_model_loader: PASSED
# - test_ab_testing: PASSED
# - test_versioning: PASSED
# - Coverage: >85%
```

### Integration Test Verification

#### Test 1: Model Registry Operations

```python
def test_model_registry_operations():
    """Verify model registry CRUD operations."""
    from rag_factory.models.embedding import ModelRegistry

    registry = ModelRegistry(registry_path="./test_registry")

    # Register model
    metadata = EmbeddingModelMetadata(
        model_id="test_model_v1",
        model_name="Test Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512
    )

    registry.register_model(metadata)

    # Retrieve model
    retrieved = registry.get_model("test_model_v1")
    assert retrieved is not None
    assert retrieved.model_name == "Test Model"

    # List models
    models = registry.list_models()
    assert len(models) == 1

    # Update metrics
    registry.update_metrics("test_model_v1", {"retrieval_accuracy": 0.85})
    updated = registry.get_model("test_model_v1")
    assert updated.retrieval_accuracy == 0.85

    # Delete model
    success = registry.delete_model("test_model_v1")
    assert success == True

    print("✓ Model registry operations verified")
```

**Expected Results:**
- All CRUD operations work correctly
- Metadata persisted to disk
- Search and filtering functional

#### Test 2: A/B Testing Framework

```python
def test_ab_testing_framework():
    """Verify A/B testing functionality."""
    from rag_factory.models.evaluation import ABTestingFramework

    framework = ABTestingFramework()

    # Start test
    config = ABTestConfig(
        test_name="base_vs_finetuned",
        model_a_id="base_model",
        model_b_id="finetuned_model",
        traffic_split=0.5,
        minimum_samples=100
    )

    framework.start_test(config)

    # Simulate requests
    for i in range(200):
        use_b = framework.should_use_model_b("base_vs_finetuned")
        model_id = "finetuned_model" if use_b else "base_model"

        # Simulate performance (B is better)
        latency = 45.0 if use_b else 50.0
        accuracy = 0.88 if use_b else 0.82

        framework.record_result(
            "base_vs_finetuned",
            model_id,
            {"latency": latency + np.random.randn() * 2,
             "accuracy": accuracy + np.random.randn() * 0.02}
        )

    # Analyze results
    result = framework.analyze_test("base_vs_finetuned")

    # Verify winner
    assert result.winner == "model_b"
    assert result.model_a_samples >= 80  # Approximately 50%
    assert result.model_b_samples >= 80

    # Verify metrics
    assert "latency" in result.metrics
    assert "accuracy" in result.metrics

    # Verify statistical significance
    assert all(p < 0.05 for p in result.p_values.values())

    print("✓ A/B testing framework verified")
```

**Expected Results:**
- Traffic splitting works (50/50)
- Metrics tracked correctly
- Statistical analysis identifies winner
- Confidence intervals calculated

#### Test 3: Model Loading and Inference

```python
def test_model_loading_and_inference():
    """Verify model loading and inference."""
    from rag_factory.models.embedding import CustomModelLoader

    loader = CustomModelLoader()

    # Load model
    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    model = loader.load_model(config)

    # Generate embeddings
    texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Python is a programming language."
    ]

    embeddings = loader.embed_texts(texts, model, config)

    # Verify embeddings
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)  # Correct dimension

    # Verify embeddings are different
    from scipy.spatial.distance import cosine
    similarity_1_2 = 1 - cosine(embeddings[0], embeddings[1])
    similarity_1_3 = 1 - cosine(embeddings[0], embeddings[2])

    # Related texts should be more similar
    assert similarity_1_2 > similarity_1_3

    # Test caching
    model2 = loader.load_model(config)
    assert model2 is model  # Should return cached model

    print("✓ Model loading and inference verified")
```

**Expected Results:**
- Model loads successfully
- Embeddings generated correctly
- Embedding dimensions correct
- Model caching works
- Inference time acceptable

### Performance Benchmarks

```bash
# Run performance benchmarks
pytest tests/benchmarks/test_model_comparison_performance.py -v

# Check outputs:
# - Model loading: <2s
# - Inference: >10 texts/second
# - A/B test overhead: <10ms per request
```

### End-to-End Workflow

```python
def test_end_to_end_workflow():
    """Test complete fine-tuned embeddings workflow."""
    # 1. Register models
    registry = ModelRegistry()

    registry.register_model(EmbeddingModelMetadata(
        model_id="base_v1",
        model_name="Base Model",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        embedding_dim=384,
        max_seq_length=512
    ))

    registry.register_model(EmbeddingModelMetadata(
        model_id="medical_v1",
        model_name="Medical Fine-tuned",
        version="1.0.0",
        format=ModelFormat.SENTENCE_TRANSFORMERS,
        base_model="base_v1",
        domain="medical",
        embedding_dim=384,
        max_seq_length=512
    ))

    # 2. Load models
    loader = CustomModelLoader()

    base_model = loader.load_model(ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS
    ))

    # 3. Start A/B test
    ab_framework = ABTestingFramework()
    ab_framework.start_test(ABTestConfig(
        test_name="base_vs_medical",
        model_a_id="base_v1",
        model_b_id="medical_v1",
        traffic_split=0.2
    ))

    # 4. Simulate production usage
    for i in range(100):
        query = f"Medical query {i}"

        use_finetuned = ab_framework.should_use_model_b("base_vs_medical")
        model_id = "medical_v1" if use_finetuned else "base_v1"

        # Generate embedding (simplified)
        embedding = loader.embed_texts([query], base_model, ModelConfig(
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            model_format=ModelFormat.SENTENCE_TRANSFORMERS
        ))[0]

        # Record metrics
        ab_framework.record_result(
            "base_vs_medical",
            model_id,
            {"latency": 45.0, "accuracy": 0.85}
        )

    # 5. Analyze and decide
    result = ab_framework.analyze_test("base_vs_medical")

    print(f"Winner: {result.winner}")
    print(f"Recommendation: {result.recommendation}")

    # 6. Gradual rollout if successful
    if result.winner == "model_b":
        ab_framework.gradual_rollout("base_vs_medical", 0.5)
        print("Increased traffic to 50%")

    print("✓ End-to-end workflow verified")
```

**Expected Results:**
- Complete workflow executes without errors
- Models registered and loaded
- A/B testing functional
- Decision-making works
- Gradual rollout successful

### Manual Verification Checklist

- [ ] Model registry stores and retrieves models correctly
- [ ] Model loading works for all formats (Hugging Face, Sentence-Transformers)
- [ ] A/B testing traffic splitting accurate
- [ ] Statistical analysis identifies winners correctly
- [ ] Model versioning and rollback functional
- [ ] Performance metrics tracked
- [ ] Documentation complete with examples

---

## Epic-Level Integration Tests

### Cross-Strategy Integration

```python
def test_all_strategies_integrated():
    """Test that all Epic 7 strategies work together."""
    # Use knowledge graph with fine-tuned embeddings
    registry = ModelRegistry()
    registry.register_model(medical_model_metadata)

    loader = CustomModelLoader()
    model = loader.load_model(medical_model_config)

    kg_strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=vector_store,
        llm_service=llm,
        config={"graph_backend": "memory"}
    )

    # Index with fine-tuned embeddings + knowledge graph
    medical_corpus = load_medical_documents()

    for doc in medical_corpus:
        # Use fine-tuned embeddings
        doc_embedding = loader.embed_texts([doc["text"]], model, medical_model_config)[0]

        # Index with KG
        kg_strategy.index_document(doc["text"], doc["id"])

    # Query
    results = kg_strategy.retrieve("What causes diabetes?", top_k=5)

    # Verify both graph and embeddings used
    assert len(results) > 0
    assert any("related_entities" in r for r in results)

    print("✓ Cross-strategy integration verified")
```

### System Performance Under Load

```python
def test_epic_7_performance_under_load():
    """Test all strategies under load."""
    import concurrent.futures
    import time

    # Setup all strategies
    strategies = {
        "knowledge_graph": setup_kg_strategy(),
        "late_chunking": setup_late_chunking_strategy(),
        "fine_tuned": setup_finetuned_strategy()
    }

    # Load test queries
    queries = generate_test_queries(n=1000)

    # Concurrent requests
    results = {}
    for strategy_name, strategy in strategies.items():
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(strategy.retrieve, query, top_k=5)
                for query in queries[:100]  # 100 queries
            ]

            responses = [f.result() for f in futures]

        duration = time.time() - start

        results[strategy_name] = {
            "total_time": duration,
            "avg_latency": duration / len(queries[:100]),
            "throughput": len(queries[:100]) / duration
        }

        print(f"{strategy_name}:")
        print(f"  Total: {duration:.2f}s")
        print(f"  Avg latency: {results[strategy_name]['avg_latency']*1000:.1f}ms")
        print(f"  Throughput: {results[strategy_name]['throughput']:.1f} queries/s")

    # Verify performance acceptable
    assert results["knowledge_graph"]["avg_latency"] < 0.5  # <500ms
    assert results["late_chunking"]["avg_latency"] < 3.0   # <3s
    assert results["fine_tuned"]["avg_latency"] < 0.1      # <100ms

    print("✓ Performance under load verified")
```

---

## Performance Validation

### Latency Requirements

| Strategy | Metric | Target | Verification |
|----------|--------|--------|--------------|
| Knowledge Graph | Entity extraction | <2s | `pytest test_kg_performance.py::test_entity_extraction_speed` |
| Knowledge Graph | Hybrid search | <500ms | `pytest test_kg_performance.py::test_hybrid_search_latency` |
| Late Chunking | Document embedding | <2s | `pytest test_late_chunking_performance.py::test_embedding_speed` |
| Late Chunking | Chunking | <300ms | `pytest test_late_chunking_performance.py::test_chunking_speed` |
| Fine-Tuned | Model loading | <2s | `pytest test_model_performance.py::test_loading_speed` |
| Fine-Tuned | Inference | >10 texts/s | `pytest test_model_performance.py::test_inference_throughput` |

### Resource Usage

```bash
# Monitor resource usage during tests
python -m memory_profiler tests/integration/test_epic_7_resources.py

# Expected limits:
# - Knowledge Graph: <4GB RAM
# - Late Chunking: <8GB RAM per document
# - Fine-Tuned: <2GB RAM per model
```

---

## Quality Metrics

### Knowledge Graph Quality

```python
def test_kg_quality_metrics():
    """Measure knowledge graph quality."""
    test_dataset = load_annotated_kg_dataset()  # 100 documents with entity annotations

    extractor = EntityExtractor(llm_service, config)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for doc in test_dataset:
        predicted = extractor.extract_entities(doc["text"], doc["id"])
        actual = doc["entities"]

        # Calculate F1
        pred_names = {e.name.lower() for e in predicted}
        actual_names = {e["name"].lower() for e in actual}

        tp = len(pred_names & actual_names)
        fp = len(pred_names - actual_names)
        fn = len(actual_names - pred_names)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Entity Extraction F1: {f1:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

    assert f1 > 0.80, f"F1 score too low: {f1:.3f} (expected >0.80)"

    print("✓ Knowledge graph quality verified")
```

### Late Chunking Quality

```python
def test_late_chunking_quality():
    """Measure late chunking quality."""
    test_docs = load_test_documents()

    late_strategy = LateChunkingRAGStrategy(vector_store, config)
    traditional_strategy = TraditionalChunkingStrategy(vector_store, config)

    late_coherence = []
    traditional_coherence = []

    for doc in test_docs:
        # Late chunking
        late_strategy.index_document(doc["text"], doc["id"])
        # Retrieve coherence from indexed chunks
        late_chunks = get_indexed_chunks(doc["id"], "late")
        late_coherence.extend([c["coherence_score"] for c in late_chunks])

        # Traditional
        traditional_strategy.index_document(doc["text"], doc["id"])
        trad_chunks = get_indexed_chunks(doc["id"], "traditional")
        traditional_coherence.extend([c["coherence_score"] for c in trad_chunks])

    # Compare
    late_avg = np.mean(late_coherence)
    trad_avg = np.mean(traditional_coherence)

    improvement = ((late_avg - trad_avg) / trad_avg) * 100

    print(f"Late Chunking Coherence: {late_avg:.3f}")
    print(f"Traditional Coherence: {trad_avg:.3f}")
    print(f"Improvement: {improvement:.1f}%")

    assert late_avg > trad_avg, "Late chunking should have higher coherence"
    assert late_avg > 0.85, f"Coherence too low: {late_avg:.3f}"

    print("✓ Late chunking quality verified")
```

### Fine-Tuned Embeddings Quality

```python
def test_finetuned_quality():
    """Measure fine-tuned embeddings quality."""
    test_dataset = load_benchmark_dataset()  # Domain-specific benchmark

    base_model = load_model("base_model_v1")
    finetuned_model = load_model("finetuned_model_v1")

    base_scores = []
    finetuned_scores = []

    for query in test_dataset:
        ground_truth = query["relevant_docs"]

        # Base model
        base_results = retrieve_with_model(query["text"], base_model)
        base_score = calculate_mrr(base_results, ground_truth)
        base_scores.append(base_score)

        # Fine-tuned model
        ft_results = retrieve_with_model(query["text"], finetuned_model)
        ft_score = calculate_mrr(ft_results, ground_truth)
        finetuned_scores.append(ft_score)

    # Compare
    base_avg = np.mean(base_scores)
    ft_avg = np.mean(finetuned_scores)

    improvement = ((ft_avg - base_avg) / base_avg) * 100

    print(f"Base Model MRR: {base_avg:.3f}")
    print(f"Fine-tuned MRR: {ft_avg:.3f}")
    print(f"Improvement: {improvement:.1f}%")

    assert improvement > 5, f"Improvement too low: {improvement:.1f}% (expected >5%)"

    print("✓ Fine-tuned embeddings quality verified")
```

---

## Final Epic Verification

### Completion Checklist

#### Story 7.1: Knowledge Graph
- [ ] All unit tests pass (>85% coverage)
- [ ] Entity extraction F1 >0.80
- [ ] Relationship extraction F1 >0.70
- [ ] Hybrid search functional
- [ ] Performance requirements met
- [ ] Documentation complete
- [ ] Code reviewed

#### Story 7.2: Late Chunking
- [ ] All unit tests pass (>85% coverage)
- [ ] Coherence scores >0.85
- [ ] Text reconstruction accurate
- [ ] Performance requirements met
- [ ] Comparison with traditional documented
- [ ] Documentation complete
- [ ] Code reviewed

#### Story 7.3: Fine-Tuned Embeddings
- [ ] All unit tests pass (>85% coverage)
- [ ] A/B testing framework functional
- [ ] Model registry working
- [ ] Accuracy improvement >5%
- [ ] Performance requirements met
- [ ] Documentation complete
- [ ] Code reviewed

### Sign-Off Criteria

Epic 7 is complete when:
1. ✅ All stories meet their individual completion criteria
2. ✅ Cross-strategy integration tests pass
3. ✅ Performance requirements met for all strategies
4. ✅ Quality metrics exceed targets
5. ✅ Documentation complete and reviewed
6. ✅ Code reviewed and approved
7. ✅ ROI analysis shows value justifies complexity

---

## Troubleshooting Common Issues

### Knowledge Graph Issues

**Issue:** Entity extraction returns poor results
- **Check:** LLM prompt quality
- **Fix:** Improve prompts, add examples, tune temperature
- **Test:** `pytest test_kg_entity_extraction.py -v`

**Issue:** Graph queries slow
- **Check:** Graph size, indexing
- **Fix:** Add indexes, optimize traversal depth
- **Test:** `pytest test_kg_performance.py::test_traversal`

### Late Chunking Issues

**Issue:** Out of memory errors
- **Check:** Document size, model size
- **Fix:** Reduce batch size, use smaller model, chunk documents
- **Test:** Monitor with `memory_profiler`

**Issue:** Text reconstruction inaccurate
- **Check:** Token boundaries, character offsets
- **Fix:** Review tokenizer offset mapping
- **Test:** `pytest test_text_reconstruction.py -v`

### Fine-Tuned Embeddings Issues

**Issue:** A/B test not showing significant results
- **Check:** Sample size, effect size
- **Fix:** Collect more data, ensure clear performance difference
- **Test:** `pytest test_ab_testing.py::test_statistical_power`

**Issue:** Model loading failures
- **Check:** Model path, format
- **Fix:** Verify model files, check format compatibility
- **Test:** `pytest test_model_loader.py -v`

---

## Contact & Support

For Epic 7 verification issues:
1. Review story-specific documentation
2. Check unit and integration test logs
3. Consult troubleshooting guide above
4. Contact development team

**Remember:** Epic 7 strategies are experimental. Thorough verification is critical before production deployment.
