# Medium RAG Factory Example - Strategy Pipeline

This example demonstrates using multiple chunking strategies and comparing their performance.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the example:
```bash
python strategy_pipeline.py
```

## What This Example Does

1. **Loads Configuration**: Reads strategy parameters from `config.yaml`
2. **Creates Multiple Chunkers**: Initializes three different chunking strategies:
   - **Structural**: Respects document structure (headers, paragraphs)
   - **Fixed-Size**: Simple token-based chunking with overlap
   - **Hybrid**: Combines structural and semantic approaches
3. **Processes Document**: Chunks the same document with all strategies
4. **Compares Performance**: Measures and displays timing for each strategy
5. **Shows Results**: Displays sample chunks and statistics

## Strategy Comparison

### Structural Chunking
- **Best for**: Markdown, HTML, structured documents
- **Pros**: Preserves logical document structure
- **Cons**: Variable chunk sizes

### Fixed-Size Chunking
- **Best for**: Plain text, consistent processing
- **Pros**: Fast, predictable chunk sizes
- **Cons**: May split semantic units

### Hybrid Chunking
- **Best for**: Mixed content types
- **Pros**: Balances structure and size
- **Cons**: More complex configuration

## Configuration

Edit `config.yaml` to customize:
- Target chunk sizes
- Overlap amounts
- Strategy-specific parameters

```yaml
chunking:
  structural:
    target_size: 256
  fixed_size:
    target_size: 200
    overlap: 20
  hybrid:
    target_size: 256
```

## Expected Output

```
Initializing Chunking Strategies
============================================================
Creating structural chunker...
Creating fixed-size chunker...
Creating hybrid chunker...

Processing Document
============================================================

Processing with structural strategy...
  ✓ Created 8 chunks in 0.003s

Processing with fixed_size strategy...
  ✓ Created 5 chunks in 0.002s

Processing with hybrid strategy...
  ✓ Created 7 chunks in 0.003s

Performance Comparison
============================================================
structural      0.003s    8 chunks  2666.7 chunks/s
fixed_size      0.002s    5 chunks  2500.0 chunks/s
hybrid          0.003s    7 chunks  2333.3 chunks/s
```

## Next Steps

- Try the [advanced example](../advanced/) for production patterns
- Experiment with different configuration values
- Add your own documents to process
- Read the [chunking strategy guide](../../docs/guides/chunking.md)

## Troubleshooting

**FileNotFoundError: config.yaml**
- Make sure you're running from the `examples/medium/` directory
- Or provide the full path to the config file

**Import errors**
- Install requirements: `pip install -r requirements.txt`
- For local development: `pip install -e ../..`

**Performance seems slow**
- This is normal for the first run (module loading)
- Subsequent runs will be faster
- Try smaller documents for testing
