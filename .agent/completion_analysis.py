#!/usr/bin/env python3
"""
Completion Analysis Script
Analyzes the ragTools project to generate completion tables for each epic.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Project root
PROJECT_ROOT = Path("/mnt/MCPProyects/ragTools")

# Epic definitions with their stories
EPICS = {
    "epic-01": {
        "name": "Core Infrastructure & Factory Pattern",
        "stories": {
            "1.1": "Design RAG Strategy Interface",
            "1.2": "Implement RAG Factory",
            "1.3": "Build Strategy Composition Engine",
            "1.4": "Create Configuration Management System",
            "1.5": "Setup Package Structure & Distribution"
        }
    },
    "epic-02": {
        "name": "Database & Storage Infrastructure",
        "stories": {
            "2.1": "Set Up Vector Database with PG Vector",
            "2.2": "Implement Database Repository Pattern"
        }
    },
    "epic-03": {
        "name": "Core Services Layer",
        "stories": {
            "3.1": "Build Embedding Service",
            "3.2": "Implement LLM Service Adapter"
        }
    },
    "epic-04": {
        "name": "Priority RAG Strategies",
        "stories": {
            "4.1": "Implement Context-Aware Chunking Strategy",
            "4.2": "Implement Re-ranking Strategy",
            "4.3": "Implement Query Expansion Strategy"
        }
    },
    "epic-05": {
        "name": "Agentic & Advanced Retrieval Strategies",
        "stories": {
            "5.1": "Implement Agentic RAG Strategy",
            "5.2": "Implement Hierarchical RAG Strategy",
            "5.3": "Implement Self-Reflective RAG Strategy"
        }
    },
    "epic-06": {
        "name": "Multi-Query & Contextual Strategies",
        "stories": {
            "6.1": "Implement Multi-Query RAG Strategy",
            "6.2": "Implement Contextual Retrieval Strategy"
        }
    },
    "epic-07": {
        "name": "Advanced & Experimental Strategies",
        "stories": {
            "7.1": "Implement Knowledge Graph Strategy",
            "7.2": "Implement Late Chunking Strategy",
            "7.3": "Implement Fine-Tuned Embeddings Strategy"
        }
    },
    "epic-08": {
        "name": "Observability & Quality Assurance",
        "stories": {
            "8.1": "Build Monitoring & Logging System",
            "8.2": "Create Evaluation Framework"
        }
    },
    "epic-08.5": {
        "name": "Development Tools (CLI & Dev Server)",
        "stories": {
            "8.5.1": "Build CLI for Strategy Testing",
            "8.5.2": "Create Lightweight Dev Server for POCs"
        }
    },
    "epic-09": {
        "name": "Documentation & Developer Experience",
        "stories": {
            "9.1": "Write Developer Documentation",
            "9.2": "Create Example Implementations"
        }
    },
    "epic-10": {
        "name": "Lightweight Dependencies Implementation",
        "stories": {
            "10.1": "Migrate Embedding Services to ONNX",
            "10.2": "Replace Tokenization with Tiktoken",
            "10.3": "Migrate Late Chunking to ONNX",
            "10.4": "Migrate Reranking to Lightweight Alternatives",
            "10.5": "Migrate Fine-Tuned Embeddings to ONNX"
        }
    },
    "epic-11": {
        "name": "Dependency Injection & Service Interface Decoupling",
        "stories": {
            "11.1": "Define Service Interfaces",
            "11.2": "Create StrategyDependencies Container",
            "11.3": "Implement Service Implementations",
            "11.4": "Update Strategy Base Classes for DI",
            "11.5": "Update RAGFactory for DI",
            "11.6": "Implement Consistency Checker"
        }
    },
    "epic-12": {
        "name": "Indexing/Retrieval Pipeline Separation",
        "stories": {
            "12.1": "Define Capability Enums and Models",
            "12.2": "Create IIndexingStrategy Interface",
            "12.3": "Create IRetrievalStrategy Interface",
            "12.4": "Implement IndexingPipeline",
            "12.5": "Implement RetrievalPipeline",
            "12.6": "Implement Factory Validation with Consistency Checking"
        }
    },
    "epic-13": {
        "name": "Core Indexing Strategies Implementation",
        "stories": {
            "13.1": "Implement Context-Aware Chunking (Indexing)",
            "13.2": "Implement Vector Embedding Indexing",
            "13.3": "Implement Keyword Extraction Indexing",
            "13.4": "Implement Hierarchical Indexing",
            "13.5": "Implement In-Memory Indexing (Testing)"
        }
    },
    "epic-14": {
        "name": "CLI Enhancements for Pipeline Validation",
        "stories": {
            "14.1": "Add Pipeline Validation Command",
            "14.2": "Add Consistency Checking Command"
        }
    },
    "epic-16": {
        "name": "Database Migration System Consolidation",
        "stories": {
            "16.1": "Audit and Document Migration Systems",
            "16.2": "Standardize Environment Variables",
            "16.3": "Create Database Connection Fixtures",
            "16.4": "Migrate Tests to Alembic",
            "16.5": "Remove Custom MigrationManager",
            "16.6": "Update Database Documentation"
        }
    },
    "epic-17": {
        "name": "Strategy Pair Configuration System",
        "stories": {
            "17.1": "Design Strategy Pair Configuration Schema",
            "17.2": "Implement StrategyPair Model and Loader",
            "17.3": "Implement Migration Validator",
            "17.4": "Implement Schema Validator",
            "17.5": "Implement StrategyPairManager",
            "17.6": "Create Pre-Built Strategy Pair Configurations"
        }
    }
}

def find_code_files(pattern: str) -> List[Path]:
    """Find code files matching a pattern."""
    code_files = []
    for ext in ['.py']:
        code_files.extend(PROJECT_ROOT.glob(f"rag_factory/**/*{pattern}*{ext}"))
    return code_files

def find_test_files(pattern: str) -> List[Path]:
    """Find test files matching a pattern."""
    test_files = []
    for ext in ['.py']:
        test_files.extend(PROJECT_ROOT.glob(f"tests/**/*{pattern}*{ext}"))
    return test_files

def find_doc_files(pattern: str) -> List[Path]:
    """Find documentation files matching a pattern."""
    doc_files = []
    for ext in ['.md']:
        doc_files.extend(PROJECT_ROOT.glob(f"docs/**/*{pattern}*{ext}"))
    return doc_files

def analyze_epic(epic_id: str, epic_data: Dict) -> List[Dict]:
    """Analyze an epic and return completion records."""
    records = []
    
    for story_id, story_name in epic_data["stories"].items():
        # Map story to features and find related files
        feature_desc = story_name[:50]  # First 50 chars as feature description
        
        # Try to find related code, tests, and docs
        # This is a simplified heuristic - you may need to adjust patterns
        search_terms = extract_search_terms(story_name)
        
        code_files = []
        test_files = []
        doc_files = []
        
        for term in search_terms:
            code_files.extend(find_code_files(term))
            test_files.extend(find_test_files(term))
            doc_files.extend(find_doc_files(term))
        
        # Deduplicate
        code_files = list(set(code_files))
        test_files = list(set(test_files))
        doc_files = list(set(doc_files))
        
        # Determine state
        state = determine_state(code_files, test_files)
        
        record = {
            "epic": epic_data["name"],
            "story": story_id,
            "state": state,
            "tests": ";".join([str(f.relative_to(PROJECT_ROOT)) for f in test_files]) if test_files else "",
            "code_files": ";".join([str(f.relative_to(PROJECT_ROOT)) for f in code_files]) if code_files else "",
            "doc_files": ";".join([str(f.relative_to(PROJECT_ROOT)) for f in doc_files]) if doc_files else "",
            "feature": feature_desc
        }
        
        records.append(record)
    
    return records

def extract_search_terms(story_name: str) -> List[str]:
    """Extract search terms from story name."""
    # Convert to lowercase and split
    words = re.findall(r'\w+', story_name.lower())
    
    # Filter out common words
    stop_words = {'implement', 'create', 'build', 'design', 'setup', 'add', 'update', 'write', 'for', 'with', 'and', 'the', 'a', 'an'}
    terms = [w for w in words if w not in stop_words and len(w) > 3]
    
    return terms[:3]  # Return top 3 terms

def determine_state(code_files: List[Path], test_files: List[Path]) -> str:
    """Determine implementation state based on files found."""
    if not code_files and not test_files:
        return "pending"
    elif code_files and test_files:
        # Check if tests are passing (simplified - would need to parse test results)
        return "code implemented"  # Could be "tests passing" if we verify
    elif code_files:
        return "code implemented"
    elif test_files:
        return "test created"
    else:
        return "pending"

def generate_completion_table():
    """Generate the completion table CSV."""
    all_records = []
    
    for epic_id, epic_data in EPICS.items():
        records = analyze_epic(epic_id, epic_data)
        all_records.extend(records)
    
    # Write to CSV
    output_file = PROJECT_ROOT / ".agent" / "completion_table.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epic", "story", "state", "tests", "code_files", "doc_files", "feature"])
        writer.writeheader()
        writer.writerows(all_records)
    
    print(f"Generated completion table: {output_file}")
    print(f"Total records: {len(all_records)}")
    
    return all_records

if __name__ == "__main__":
    records = generate_completion_table()
    
    # Print summary
    states = {}
    for record in records:
        state = record["state"]
        states[state] = states.get(state, 0) + 1
    
    print("\nState Summary:")
    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")
