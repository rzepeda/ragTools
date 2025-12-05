"""
Dataset schema definitions for evaluation.

This module defines the data structures for evaluation datasets,
including examples and complete datasets.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class EvaluationExample:
    """
    Single evaluation example.

    Represents one query-answer pair with relevance judgments
    for evaluation purposes.

    Attributes:
        query_id: Unique identifier for the query
        query: The query text
        ground_truth_answer: Optional ground truth answer for quality metrics
        relevant_doc_ids: Set of IDs for documents relevant to this query
        relevance_scores: Optional graded relevance scores (doc_id -> score)
        metadata: Additional metadata for the example

    Example:
        >>> example = EvaluationExample(
        ...     query_id="q1",
        ...     query="What is machine learning?",
        ...     ground_truth_answer="Machine learning is...",
        ...     relevant_doc_ids={"doc1", "doc2"}
        ... )
    """
    query_id: str
    query: str
    ground_truth_answer: Optional[str] = None
    relevant_doc_ids: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate example after initialization."""
        if not self.query_id:
            raise ValueError("query_id cannot be empty")
        if not self.query:
            raise ValueError("query cannot be empty")
        # Ensure relevant_doc_ids is a set
        if not isinstance(self.relevant_doc_ids, set):
            self.relevant_doc_ids = set(self.relevant_doc_ids)


@dataclass
class EvaluationDataset:
    """
    Collection of evaluation examples.

    Represents a complete evaluation dataset with multiple query-answer pairs.

    Attributes:
        name: Dataset name/identifier
        examples: List of evaluation examples
        metadata: Additional dataset-level metadata

    Example:
        >>> dataset = EvaluationDataset(
        ...     name="test_dataset",
        ...     examples=[example1, example2, example3]
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> first_example = dataset[0]
    """
    name: str
    examples: List[EvaluationExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> EvaluationExample:
        """Get example by index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def get_by_id(self, query_id: str) -> Optional[EvaluationExample]:
        """
        Get example by query ID.

        Args:
            query_id: Query identifier

        Returns:
            EvaluationExample if found, None otherwise
        """
        for example in self.examples:
            if example.query_id == query_id:
                return example
        return None

    def split(self, train_ratio: float = 0.8, seed: Optional[int] = None):
        """
        Split dataset into train and test sets.

        Args:
            train_ratio: Proportion of data for training (default: 0.8)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random
        if seed is not None:
            random.seed(seed)

        examples_copy = self.examples.copy()
        random.shuffle(examples_copy)

        split_idx = int(len(examples_copy) * train_ratio)
        train_examples = examples_copy[:split_idx]
        test_examples = examples_copy[split_idx:]

        train_dataset = EvaluationDataset(
            name=f"{self.name}_train",
            examples=train_examples,
            metadata={**self.metadata, "split": "train", "seed": seed}
        )

        test_dataset = EvaluationDataset(
            name=f"{self.name}_test",
            examples=test_examples,
            metadata={**self.metadata, "split": "test", "seed": seed}
        )

        return train_dataset, test_dataset

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.examples:
            return {
                "total_examples": 0,
                "avg_relevant_docs": 0,
                "examples_with_ground_truth": 0,
                "examples_with_relevance_scores": 0
            }

        total_relevant = sum(len(ex.relevant_doc_ids) for ex in self.examples)
        examples_with_gt = sum(1 for ex in self.examples if ex.ground_truth_answer)
        examples_with_scores = sum(1 for ex in self.examples if ex.relevance_scores)

        return {
            "total_examples": len(self.examples),
            "avg_relevant_docs": total_relevant / len(self.examples),
            "examples_with_ground_truth": examples_with_gt,
            "examples_with_relevance_scores": examples_with_scores,
            "metadata": self.metadata
        }
