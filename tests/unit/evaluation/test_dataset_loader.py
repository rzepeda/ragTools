"""
Unit tests for dataset loader.

Tests dataset loading from JSON, JSONL, and CSV formats.
"""

import pytest
import json
import csv
from pathlib import Path
from rag_factory.evaluation.datasets.loader import DatasetLoader
from rag_factory.evaluation.datasets.schema import EvaluationDataset


@pytest.fixture
def temp_json_dataset(tmp_path):
    """Create a temporary JSON dataset file."""
    data = {
        "name": "test_dataset",
        "metadata": {"version": "1.0"},
        "examples": [
            {
                "query_id": "q1",
                "query": "What is machine learning?",
                "ground_truth_answer": "ML is...",
                "relevant_doc_ids": ["doc1", "doc2"],
                "relevance_scores": {"doc1": 3, "doc2": 2}
            },
            {
                "query_id": "q2",
                "query": "What is AI?",
                "ground_truth_answer": "AI is...",
                "relevant_doc_ids": ["doc3"]
            }
        ]
    }

    path = tmp_path / "dataset.json"
    with open(path, 'w') as f:
        json.dump(data, f)

    return path


@pytest.fixture
def temp_jsonl_dataset(tmp_path):
    """Create a temporary JSONL dataset file."""
    path = tmp_path / "dataset.jsonl"

    with open(path, 'w') as f:
        f.write(json.dumps({
            "query_id": "q1",
            "query": "Test query 1",
            "relevant_doc_ids": ["doc1"]
        }) + '\n')
        f.write(json.dumps({
            "query_id": "q2",
            "query": "Test query 2",
            "relevant_doc_ids": ["doc2", "doc3"]
        }) + '\n')

    return path


@pytest.fixture
def temp_csv_dataset(tmp_path):
    """Create a temporary CSV dataset file."""
    path = tmp_path / "dataset.csv"

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'query', 'ground_truth_answer', 'relevant_doc_ids'])
        writer.writerow(['q1', 'Query 1', 'Answer 1', 'doc1,doc2'])
        writer.writerow(['q2', 'Query 2', 'Answer 2', 'doc3'])

    return path


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_load_json_dataset(self, temp_json_dataset):
        """Test loading JSON dataset."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "test_dataset"
        assert len(dataset) == 2
        assert dataset.metadata["version"] == "1.0"

        # Check first example
        ex1 = dataset[0]
        assert ex1.query_id == "q1"
        assert ex1.query == "What is machine learning?"
        assert ex1.ground_truth_answer == "ML is..."
        assert "doc1" in ex1.relevant_doc_ids
        assert "doc2" in ex1.relevant_doc_ids
        assert ex1.relevance_scores["doc1"] == 3

    def test_load_jsonl_dataset(self, temp_jsonl_dataset):
        """Test loading JSONL dataset."""
        loader = DatasetLoader()
        dataset = loader.load(temp_jsonl_dataset)

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset) == 2

        # Check examples
        ex1 = dataset[0]
        assert ex1.query_id == "q1"
        assert ex1.query == "Test query 1"

        ex2 = dataset[1]
        assert ex2.query_id == "q2"
        assert len(ex2.relevant_doc_ids) == 2

    def test_load_csv_dataset(self, temp_csv_dataset):
        """Test loading CSV dataset."""
        loader = DatasetLoader()
        dataset = loader.load(temp_csv_dataset)

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset) == 2

        # Check first example
        ex1 = dataset[0]
        assert ex1.query_id == "q1"
        assert ex1.query == "Query 1"
        assert ex1.ground_truth_answer == "Answer 1"
        assert "doc1" in ex1.relevant_doc_ids
        assert "doc2" in ex1.relevant_doc_ids

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = DatasetLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.json")

    def test_load_unsupported_format(self, tmp_path):
        """Test loading unsupported format raises error."""
        path = tmp_path / "data.txt"
        path.write_text("test")

        loader = DatasetLoader()

        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(path)

    def test_dataset_name_override(self, temp_json_dataset):
        """Test overriding dataset name."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset, dataset_name="custom_name")

        assert dataset.name == "custom_name"

    def test_save_and_load_json(self, tmp_path, temp_json_dataset):
        """Test saving and loading JSON dataset."""
        loader = DatasetLoader()

        # Load original
        original = loader.load(temp_json_dataset)

        # Save to new path
        new_path = tmp_path / "saved.json"
        loader.save(original, new_path, format="json")

        # Load saved dataset
        loaded = loader.load(new_path)

        assert len(loaded) == len(original)
        assert loaded.name == original.name

    def test_save_and_load_jsonl(self, tmp_path, temp_json_dataset):
        """Test saving and loading JSONL dataset."""
        loader = DatasetLoader()

        # Load original
        original = loader.load(temp_json_dataset)

        # Save as JSONL
        new_path = tmp_path / "saved.jsonl"
        loader.save(original, new_path, format="jsonl")

        # Load saved dataset
        loaded = loader.load(new_path)

        assert len(loaded) == len(original)

    def test_save_and_load_csv(self, tmp_path, temp_json_dataset):
        """Test saving and loading CSV dataset."""
        loader = DatasetLoader()

        # Load original
        original = loader.load(temp_json_dataset)

        # Save as CSV
        new_path = tmp_path / "saved.csv"
        loader.save(original, new_path, format="csv")

        # Load saved dataset
        loaded = loader.load(new_path)

        assert len(loaded) == len(original)


class TestEvaluationDataset:
    """Tests for EvaluationDataset class."""

    def test_dataset_iteration(self, temp_json_dataset):
        """Test iterating over dataset."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        count = 0
        for example in dataset:
            count += 1
            assert example.query_id is not None

        assert count == len(dataset)

    def test_dataset_indexing(self, temp_json_dataset):
        """Test accessing examples by index."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        first = dataset[0]
        second = dataset[1]

        assert first.query_id == "q1"
        assert second.query_id == "q2"

    def test_get_by_id(self, temp_json_dataset):
        """Test getting example by query ID."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        example = dataset.get_by_id("q1")
        assert example is not None
        assert example.query_id == "q1"

        non_existent = dataset.get_by_id("q999")
        assert non_existent is None

    def test_dataset_split(self, temp_json_dataset):
        """Test splitting dataset into train/test."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        train, test = dataset.split(train_ratio=0.5, seed=42)

        # Check split
        assert len(train) + len(test) == len(dataset)
        assert train.name.endswith("_train")
        assert test.name.endswith("_test")

    def test_get_statistics(self, temp_json_dataset):
        """Test getting dataset statistics."""
        loader = DatasetLoader()
        dataset = loader.load(temp_json_dataset)

        stats = dataset.get_statistics()

        assert stats["total_examples"] == 2
        assert stats["examples_with_ground_truth"] == 2
        assert stats["avg_relevant_docs"] > 0
