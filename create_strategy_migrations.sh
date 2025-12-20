#!/bin/bash
# Script to create all missing strategy migrations

set -e

cd /mnt/MCPProyects/ragTools
source venv/bin/activate

echo "Creating missing strategy migrations..."
echo "========================================"

# List of migrations to create (excluding already existing ones)
# Already have: 001, 002, semantic_local_schema, agentic_schema, context_aware_schema

migrations=(
    "contextual_schema"
    "finetuned_schema"
    "hierarchical_schema"
    "keyword_schema"
    "knowledge_graph_schema"
    "late_chunking_schema"
    "multi_query_schema"
    "query_expansion_schema"
    "reranking_schema"
    "self_reflective_schema"
    "semantic_api_schema"
)

for migration in "${migrations[@]}"; do
    echo "Creating migration: $migration"
    alembic revision -m "$migration" --rev-id="$migration" --head="agentic_schema" --splice
done

echo ""
echo "✅ All migration files created!"
echo "Now running: alembic upgrade head to apply all migrations"
alembic upgrade head
