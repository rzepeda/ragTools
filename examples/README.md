# RAG Factory CLI Examples

This directory contains example files for using the RAG Factory CLI.

## Files

### Configuration Examples

#### `cli_config_example.yaml`
Sample YAML configuration file demonstrating various CLI options:
- Strategy selection
- Chunking parameters (size, overlap)
- Retrieval settings (top-k)
- Strategy-specific settings
- Metadata fields

**Usage:**
```bash
# Validate the configuration
rag-factory config cli_config_example.yaml

# Use with index command
rag-factory index ./docs --config cli_config_example.yaml

# Use with query command
rag-factory query "test" --config cli_config_example.yaml
```

### Benchmark Examples

#### `benchmark_dataset_example.json`
Sample benchmark dataset showing the expected format for benchmark queries:
- Query text
- Expected relevant documents
- Metadata (category, difficulty, expected scores)

**Usage:**
```bash
# Run benchmark
rag-factory benchmark benchmark_dataset_example.json

# Benchmark specific strategies
rag-factory benchmark benchmark_dataset_example.json --strategies reranking,semantic_chunker

# Export results
rag-factory benchmark benchmark_dataset_example.json --output results.json
```

## Creating Your Own Configuration

### YAML Configuration Template

```yaml
# my_config.yaml
strategy_name: your_strategy_name  # Required

# Optional parameters
chunk_size: 512
chunk_overlap: 50
top_k: 10

# Add strategy-specific settings here
# ...
```

### JSON Configuration Template

```json
{
  "strategy_name": "your_strategy_name",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 10
}
```

## Creating Benchmark Datasets

Benchmark datasets should be JSON arrays of query objects:

```json
[
  {
    "query": "Your query text here",
    "expected_docs": ["doc1.txt", "doc2.txt"],
    "metadata": {
      "category": "your_category",
      "difficulty": "easy|medium|hard",
      "expected_score": 0.85
    }
  }
]
```

### Required Fields
- `query` (string): The query text to execute

### Optional Fields
- `expected_docs` (array): List of document filenames expected to be relevant
- `metadata` (object): Additional information about the query
  - `category` (string): Query category for analysis
  - `difficulty` (string): Expected difficulty level
  - `expected_score` (number): Expected relevance score

## Quick Start Examples

### Example 1: Basic Indexing and Querying

```bash
# Create some test documents
mkdir -p test_docs
echo "Machine learning is a subset of AI." > test_docs/ml.txt
echo "Deep learning uses neural networks." > test_docs/dl.txt

# Index the documents
rag-factory index test_docs

# Query the indexed documents
rag-factory query "What is machine learning?"
```

### Example 2: Using Configuration Files

```bash
# Copy example config and modify as needed
cp cli_config_example.yaml my_config.yaml

# Validate your config
rag-factory config my_config.yaml

# Use your config
rag-factory index test_docs --config my_config.yaml
rag-factory query "neural networks" --config my_config.yaml
```

### Example 3: Running Benchmarks

```bash
# Create benchmark dataset
cat > my_benchmark.json << 'EOF'
[
  {
    "query": "What is machine learning?",
    "expected_docs": ["ml.txt"],
    "metadata": {"category": "definition"}
  },
  {
    "query": "Explain neural networks",
    "expected_docs": ["dl.txt"],
    "metadata": {"category": "technical"}
  }
]
EOF

# Run benchmark
rag-factory benchmark my_benchmark.json --output results.json

# View results
cat results.json
```

### Example 4: Interactive REPL Session

```bash
# Start REPL with configuration
rag-factory repl --config my_config.yaml

# In the REPL:
rag-factory> strategies
rag-factory> index test_docs
rag-factory> query "machine learning"
rag-factory> set strategy semantic_chunker
rag-factory> show
rag-factory> exit
```

## Tips and Best Practices

1. **Start Simple:** Begin with default settings before customizing
2. **Validate Configs:** Always validate configuration files before use
3. **Test Strategies:** Use benchmarks to compare different strategies
4. **Use REPL:** Interactive mode is great for experimentation
5. **Check Documentation:** Run `rag-factory COMMAND --help` for detailed options

## Troubleshooting

### Configuration Issues

If you get validation errors:
```bash
# Check config syntax
rag-factory config my_config.yaml --show

# Use strict mode to catch warnings
rag-factory config my_config.yaml --strict
```

### Index Not Found

If queries fail with "index not found":
```bash
# Make sure you've indexed first
rag-factory index ./docs

# Or specify the index location
rag-factory query "test" --index /path/to/index
```

## More Information

- See the full [CLI User Guide](../docs/CLI-USER-GUIDE.md)
- Read the [story documentation](../docs/stories/epic-08.5/story-8.5.1-cli-strategy-testing.md)
- Check the [completion summary](../docs/stories/epic-08.5/story-8.5.1-COMPLETION-SUMMARY.md)
