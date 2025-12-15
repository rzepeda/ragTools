# Story 17.8: End-to-End CLI Validation with Sample Documents

**As a** developer  
**I want** to index 3 sample documents and retrieve information using the CLI  
**So that** I can validate the complete system works end-to-end with minimal setup

## Acceptance Criteria
- Create 3 sample documents with known content (for testing retrieval)
- Create CLI configuration file (`cli-config.yaml`)
- CLI reads configuration from YAML file (not command-line args)
- Use `semantic-local-pair` from Story 17.6 (no API keys needed)
- Index all 3 documents via CLI
- Perform 5 test queries via CLI
- Verify correct documents are retrieved for each query
- Document the complete workflow in a tutorial
- Create automated test script that validates the workflow
- Measure and document performance metrics

## Sample Documents

**sample-docs/python_basics.txt:**
```
Python Programming Basics
...
```

**sample-docs/machine_learning.txt:**
```
Introduction to Machine Learning
...
```

**sample-docs/embeddings_explained.txt:**
```
Understanding Embeddings
...
```

## CLI Configuration File
```yaml
# CLI Configuration for RAG Factory
strategy_pair: "semantic-local-pair"
service_registry: "config/services.yaml"
alembic_config: "alembic.ini"

defaults:
  top_k: 5
  verbose: true
  validate_migrations: true

output:
  format: "table"
  show_scores: true
  show_metadata: true
  max_content_length: 200

performance:
  batch_size: 32
  show_timing: true
```

## Automated Validation Script
`tests/e2e/test_cli_workflow.sh`

## Story Points
5
