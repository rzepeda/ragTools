"""
Dataset loader for various formats.

This module provides functionality to load evaluation datasets
from JSON, JSONL, and CSV formats.
"""

import json
import csv
from pathlib import Path
from typing import Union, Dict, Any
from rag_factory.evaluation.datasets.schema import EvaluationDataset, EvaluationExample


class DatasetLoader:
    """
    Load evaluation datasets from various formats.

    Supports: JSON, JSONL, CSV

    Example:
        >>> loader = DatasetLoader()
        >>> dataset = loader.load("path/to/dataset.json")
        >>> print(f"Loaded {len(dataset)} examples")
    """

    def load(self, path: Union[str, Path], dataset_name: Optional[str] = None) -> EvaluationDataset:
        """
        Load dataset from file.

        Args:
            path: Path to dataset file
            dataset_name: Optional name override for the dataset

        Returns:
            EvaluationDataset with loaded examples

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if path.suffix == '.json':
            dataset = self._load_json(path)
        elif path.suffix == '.jsonl':
            dataset = self._load_jsonl(path)
        elif path.suffix == '.csv':
            dataset = self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: .json, .jsonl, .csv")

        # Override name if provided
        if dataset_name:
            dataset.name = dataset_name

        return dataset

    def _load_json(self, path: Path) -> EvaluationDataset:
        """
        Load from JSON file.

        Expected format:
        {
            "name": "dataset_name",
            "metadata": {...},
            "examples": [
                {
                    "query_id": "q1",
                    "query": "...",
                    "ground_truth_answer": "...",
                    "relevant_doc_ids": ["doc1", "doc2"],
                    "relevance_scores": {"doc1": 3, "doc2": 2},
                    "metadata": {...}
                }
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            EvaluationDataset
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        for i, ex in enumerate(data.get("examples", [])):
            try:
                example = EvaluationExample(
                    query_id=ex.get("query_id", f"q_{i}"),
                    query=ex["query"],
                    ground_truth_answer=ex.get("ground_truth_answer"),
                    relevant_doc_ids=set(ex.get("relevant_doc_ids", [])),
                    relevance_scores=ex.get("relevance_scores", {}),
                    metadata=ex.get("metadata", {})
                )
                examples.append(example)
            except KeyError as e:
                raise ValueError(f"Missing required field in example {i}: {e}")

        return EvaluationDataset(
            name=data.get("name", path.stem),
            examples=examples,
            metadata=data.get("metadata", {})
        )

    def _load_jsonl(self, path: Path) -> EvaluationDataset:
        """
        Load from JSONL file (one JSON object per line).

        Each line should be a JSON object with fields:
        - query_id (optional, will be generated)
        - query (required)
        - ground_truth_answer (optional)
        - relevant_doc_ids (optional)
        - relevance_scores (optional)
        - metadata (optional)

        Args:
            path: Path to JSONL file

        Returns:
            EvaluationDataset
        """
        examples = []

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    ex = json.loads(line)
                    example = EvaluationExample(
                        query_id=ex.get("query_id", f"q_{i}"),
                        query=ex["query"],
                        ground_truth_answer=ex.get("ground_truth_answer"),
                        relevant_doc_ids=set(ex.get("relevant_doc_ids", [])),
                        relevance_scores=ex.get("relevance_scores", {}),
                        metadata=ex.get("metadata", {})
                    )
                    examples.append(example)
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"Error parsing line {i+1}: {e}")

        return EvaluationDataset(
            name=path.stem,
            examples=examples
        )

    def _load_csv(self, path: Path) -> EvaluationDataset:
        """
        Load from CSV file.

        Expected columns:
        - query_id (optional)
        - query (required)
        - ground_truth_answer (optional)
        - relevant_doc_ids (comma-separated list, optional)

        Args:
            path: Path to CSV file

        Returns:
            EvaluationDataset
        """
        examples = []

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if 'query' not in reader.fieldnames:
                raise ValueError("CSV must contain 'query' column")

            for i, row in enumerate(reader):
                try:
                    # Parse relevant doc IDs from comma-separated string
                    relevant_ids_str = row.get("relevant_doc_ids", "")
                    if relevant_ids_str:
                        relevant_ids = set(
                            id.strip()
                            for id in relevant_ids_str.split(",")
                            if id.strip()
                        )
                    else:
                        relevant_ids = set()

                    example = EvaluationExample(
                        query_id=row.get("query_id", f"q_{i}"),
                        query=row["query"],
                        ground_truth_answer=row.get("ground_truth_answer"),
                        relevant_doc_ids=relevant_ids
                    )
                    examples.append(example)
                except KeyError as e:
                    raise ValueError(f"Missing required field in row {i+1}: {e}")

        return EvaluationDataset(
            name=path.stem,
            examples=examples
        )

    def save(
        self,
        dataset: EvaluationDataset,
        path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save dataset to file.

        Args:
            dataset: Dataset to save
            path: Output file path
            format: Output format ("json", "jsonl", or "csv")

        Raises:
            ValueError: If format is not supported
        """
        path = Path(path)
        format = format.lower()

        if format == "json":
            self._save_json(dataset, path)
        elif format == "jsonl":
            self._save_jsonl(dataset, path)
        elif format == "csv":
            self._save_csv(dataset, path)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: json, jsonl, csv")

    def _save_json(self, dataset: EvaluationDataset, path: Path) -> None:
        """Save dataset as JSON."""
        data = {
            "name": dataset.name,
            "metadata": dataset.metadata,
            "examples": [
                {
                    "query_id": ex.query_id,
                    "query": ex.query,
                    "ground_truth_answer": ex.ground_truth_answer,
                    "relevant_doc_ids": list(ex.relevant_doc_ids),
                    "relevance_scores": ex.relevance_scores,
                    "metadata": ex.metadata
                }
                for ex in dataset.examples
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_jsonl(self, dataset: EvaluationDataset, path: Path) -> None:
        """Save dataset as JSONL."""
        with open(path, 'w', encoding='utf-8') as f:
            for ex in dataset.examples:
                data = {
                    "query_id": ex.query_id,
                    "query": ex.query,
                    "ground_truth_answer": ex.ground_truth_answer,
                    "relevant_doc_ids": list(ex.relevant_doc_ids),
                    "relevance_scores": ex.relevance_scores,
                    "metadata": ex.metadata
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _save_csv(self, dataset: EvaluationDataset, path: Path) -> None:
        """Save dataset as CSV."""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['query_id', 'query', 'ground_truth_answer', 'relevant_doc_ids']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ex in dataset.examples:
                writer.writerow({
                    'query_id': ex.query_id,
                    'query': ex.query,
                    'ground_truth_answer': ex.ground_truth_answer or '',
                    'relevant_doc_ids': ','.join(ex.relevant_doc_ids)
                })
