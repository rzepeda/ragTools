# RAG Factory CLI User Guide

The RAG Factory CLI provides a command-line interface for testing and experimenting with RAG strategies without writing code. This guide covers installation, usage, and common workflows.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands](#commands)
4. [Configuration](#configuration)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Install from Source

```bash
# Clone the repository
git clone #.git
cd rag-factory

# Install with CLI dependencies
pip install -e ".[cli]"

# Verify installation
rag-factory --version
```

### Install CLI Dependencies Only

If you already have the library installed:

```bash
pip install typer>=0.12.0 rich>=13.7.0 prompt-toolkit>=3.0.43
```

---

## Quick Start

### 1. Index Documents

Index a directory of documents:

```bash
rag-factory index ./docs --strategy fixed_size_chunker
```

### 2. Query Indexed Documents

Query the indexed documents:

```bash
rag-factory query "What are the main topics?"
```

### 3. List Available Strategies

See all available strategies:

```bash
rag-factory strategies
```

---

## Commands

### `index` - Index Documents

Index documents using a specified chunking strategy.

**Usage:**
```bash
rag-factory index PATH [OPTIONS]
```

**Arguments:**
- `PATH` - Path to file or directory containing documents

**Options:**
- `--strategy, -s TEXT` - Chunking strategy to use (default: fixed_size_chunker)
- `--config, -c PATH` - Path to configuration file (YAML or JSON)
- `--output, -o PATH` - Directory to store indexed data (default: ./rag_index)
- `--chunk-size INTEGER` - Size of chunks in tokens (default: 512)
- `--chunk-overlap INTEGER` - Overlap between chunks (default: 50)

**Examples:**

Index a single file:
```bash
rag-factory index document.txt
```

Index a directory with custom chunk size:
```bash
rag-factory index ./docs --chunk-size 1024 --chunk-overlap 100
```

Index with a configuration file:
```bash
rag-factory index ./docs --config config.yaml
```

---

### `query` - Query Indexed Documents

Search indexed documents using specified strategies.

**Usage:**
```bash
rag-factory query QUERY [OPTIONS]
```

**Arguments:**
- `QUERY` - Query string to search for

**Options:**
- `--strategies, -s TEXT` - Comma-separated strategies (default: basic)
- `--top-k, -k INTEGER` - Number of results to return (default: 5)
- `--config, -c PATH` - Path to configuration file
- `--index, -i PATH` - Index directory (default: ./rag_index)
- `--show-scores/--no-scores` - Show relevance scores (default: True)

**Examples:**

Simple query:
```bash
rag-factory query "What is machine learning?"
```

Query with multiple strategies:
```bash
rag-factory query "neural networks" --strategies reranking,query_expansion
```

Get more results:
```bash
rag-factory query "deep learning" --top-k 10
```

---

### `strategies` - List Available Strategies

Display all registered strategies with descriptions.

**Usage:**
```bash
rag-factory strategies [OPTIONS]
```

**Options:**
- `--type, -t TEXT` - Filter by type (chunking, reranking, query_expansion)
- `--verbose, -v` - Show detailed information

**Examples:**

List all strategies:
```bash
rag-factory strategies
```

List only chunking strategies:
```bash
rag-factory strategies --type chunking
```

Show detailed information:
```bash
rag-factory strategies --verbose
```

---

### `config` - Validate Configuration

Validate a configuration file for correctness.

**Usage:**
```bash
rag-factory config PATH [OPTIONS]
```

**Arguments:**
- `PATH` - Path to configuration file

**Options:**
- `--show, -s` - Display configuration contents
- `--strict` - Treat warnings as errors

**Examples:**

Validate configuration:
```bash
rag-factory config config.yaml
```

Validate and show contents:
```bash
rag-factory config config.yaml --show
```

Strict validation:
```bash
rag-factory config config.yaml --strict
```

---

### `benchmark` - Run Benchmarks

Execute benchmarks using a test dataset.

**Usage:**
```bash
rag-factory benchmark DATASET [OPTIONS]
```

**Arguments:**
- `DATASET` - Path to benchmark dataset (JSON)

**Options:**
- `--strategies, -s TEXT` - Strategies to benchmark (default: all)
- `--output, -o PATH` - Export results to file (JSON or CSV)
- `--index, -i PATH` - Index directory (default: ./rag_index)
- `--iterations, -n INTEGER` - Iterations per query (default: 1)

**Dataset Format:**
```json
[
  {
    "query": "What is RAG?",
    "expected_docs": ["doc1.txt", "doc2.txt"],
    "metadata": {"category": "general"}
  }
]
```

**Examples:**

Run benchmark:
```bash
rag-factory benchmark queries.json
```

Benchmark specific strategies:
```bash
rag-factory benchmark queries.json --strategies reranking,semantic_chunker
```

Export results:
```bash
rag-factory benchmark queries.json --output results.json
```

Run multiple iterations:
```bash
rag-factory benchmark queries.json --iterations 5
```

---

### `repl` - Interactive Mode

Start an interactive REPL session.

**Usage:**
```bash
rag-factory repl [OPTIONS]
```

**Options:**
- `--config, -c PATH` - Configuration file to load

**REPL Commands:**
- `index <path>` - Index documents
- `query <text>` - Query indexed documents
- `strategies` - List available strategies
- `config <path>` - Load configuration
- `set <key> <value>` - Set session parameter
- `show` - Show current session state
- `help` - Show help message
- `exit/quit` - Exit REPL

**Examples:**

Start REPL:
```bash
rag-factory repl
```

Start with configuration:
```bash
rag-factory repl --config config.yaml
```

**REPL Session Example:**
```
rag-factory> index ./docs
Successfully indexed documents from ./docs

rag-factory> query "machine learning"
Results:
  1. Sample result (score: 0.95)
  2. Another result (score: 0.87)

rag-factory> set strategy semantic_chunker
Strategy set to: semantic_chunker

rag-factory> show
Current Session State:
  Strategy:     semantic_chunker
  Index Dir:    ./rag_index
  Config File:  None

rag-factory> exit
Goodbye!
```

---

## Configuration

Configuration files can be in YAML or JSON format.

### YAML Configuration Example

```yaml
# config.yaml
strategy_name: semantic_chunker
chunk_size: 512
chunk_overlap: 50
top_k: 10

# Strategy-specific settings
embedding_model: all-MiniLM-L6-v2
similarity_threshold: 0.85
```

### JSON Configuration Example

```json
{
  "strategy_name": "semantic_chunker",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 10,
  "embedding_model": "all-MiniLM-L6-v2",
  "similarity_threshold": 0.85
}
```

### Configuration Fields

**Required:**
- `strategy_name` (string) - Name of the strategy to use

**Optional:**
- `chunk_size` (integer) - Size of chunks in tokens (default: 512)
- `chunk_overlap` (integer) - Overlap between chunks (default: 50)
- `top_k` (integer) - Number of results to return (default: 5)

Additional fields depend on the specific strategy being used.

---

## Examples

### Example 1: Quick Document Indexing

Index a directory and query it:

```bash
# Index documents
rag-factory index ./my_docs

# Query for specific information
rag-factory query "What are the action items?"
```

### Example 2: Testing Different Strategies

Compare different chunking strategies:

```bash
# Index with fixed-size chunking
rag-factory index ./docs --strategy fixed_size_chunker --output ./index_fixed

# Index with semantic chunking
rag-factory index ./docs --strategy semantic_chunker --output ./index_semantic

# Query both indexes
rag-factory query "machine learning" --index ./index_fixed
rag-factory query "machine learning" --index ./index_semantic
```

### Example 3: Configuration-Based Workflow

Create a configuration file and use it:

```yaml
# my_config.yaml
strategy_name: semantic_chunker
chunk_size: 1024
chunk_overlap: 100
top_k: 20
```

```bash
# Validate configuration
rag-factory config my_config.yaml

# Use configuration for indexing
rag-factory index ./docs --config my_config.yaml

# Use configuration for querying
rag-factory query "deep learning" --config my_config.yaml
```

### Example 4: Benchmarking Strategies

Create a benchmark dataset and run tests:

```json
// benchmark_queries.json
[
  {
    "query": "What is machine learning?",
    "expected_docs": ["ml_intro.txt"],
    "metadata": {"difficulty": "easy"}
  },
  {
    "query": "Explain transformer architecture",
    "expected_docs": ["transformers.txt", "attention.txt"],
    "metadata": {"difficulty": "hard"}
  }
]
```

```bash
# Run benchmark
rag-factory benchmark benchmark_queries.json --output results.json

# View results
cat results.json
```

### Example 5: Interactive Experimentation

Use REPL for iterative development:

```bash
rag-factory repl

# Inside REPL:
rag-factory> strategies
# ... view available strategies ...

rag-factory> set strategy semantic_chunker
Strategy set to: semantic_chunker

rag-factory> index ./docs
Successfully indexed documents

rag-factory> query "What are the main concepts?"
# ... view results ...

rag-factory> set strategy fixed_size_chunker
rag-factory> index ./docs
# ... compare results with different strategy ...
```

---

## Troubleshooting

### Issue: Command not found

**Problem:** `rag-factory: command not found`

**Solution:**
```bash
# Ensure CLI is installed
pip install -e ".[cli]"

# Or run via Python module
python -m rag_factory.cli.main --help
```

### Issue: No strategies found

**Problem:** `No strategies registered`

**Solution:**
- Ensure the library is properly installed
- Check that strategy modules are loaded
- Verify import paths in your setup

### Issue: Index not found

**Problem:** `Index directory not found`

**Solution:**
```bash
# Create index first
rag-factory index ./docs

# Or specify index location
rag-factory query "test" --index /path/to/index
```

### Issue: Configuration validation errors

**Problem:** `Invalid YAML in configuration file`

**Solution:**
- Check YAML syntax (indentation, colons, etc.)
- Use `--show` flag to see parsed content
- Validate with online YAML validator

### Issue: Out of memory during indexing

**Problem:** Memory error when indexing large document sets

**Solution:**
- Reduce `chunk_size` in configuration
- Process documents in batches
- Use smaller embedding models

---

## Advanced Usage

### Debugging

Enable debug mode for verbose output:

```bash
rag-factory --debug index ./docs
```

### Custom Output Formats

Export benchmark results in different formats:

```bash
# Export to JSON
rag-factory benchmark queries.json --output results.json

# Export to CSV
rag-factory benchmark queries.json --output results.csv
```

### Scripting with the CLI

Use the CLI in bash scripts:

```bash
#!/bin/bash

# Batch processing script
for dir in ./data/*/; do
    echo "Processing $dir"
    rag-factory index "$dir" --output "./indexes/$(basename $dir)"
done

# Query all indexes
for index in ./indexes/*/; do
    echo "Querying $index"
    rag-factory query "machine learning" --index "$index"
done
```

---

## Getting Help

For command-specific help:

```bash
rag-factory COMMAND --help
```

Example:
```bash
rag-factory index --help
rag-factory query --help
```

For general help:
```bash
rag-factory --help
```

---

## Next Steps

- Read the <!-- BROKEN LINK: API Documentation <!-- (broken link to: ./API.md) --> --> API Documentation for programmatic usage
- Explore <!-- BROKEN LINK: Strategy Development Guide <!-- (broken link to: ./STRATEGIES.md) --> --> Strategy Development Guide to create custom strategies
- See [Examples](../examples/) for more use cases
- Report issues on [GitHub Issues](#/issues)
