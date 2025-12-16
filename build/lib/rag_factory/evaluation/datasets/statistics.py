"""
Dataset statistics and analysis tools.

This module provides utilities for analyzing evaluation datasets
and generating statistics.
"""

from typing import Dict, Any, List
from collections import Counter
from rag_factory.evaluation.datasets.schema import EvaluationDataset


class DatasetStatistics:
    """
    Compute and display dataset statistics.

    Example:
        >>> stats = DatasetStatistics(dataset)
        >>> summary = stats.compute()
        >>> print(stats.format_summary(summary))
    """

    def __init__(self, dataset: EvaluationDataset):
        """
        Initialize statistics computer.

        Args:
            dataset: Evaluation dataset to analyze
        """
        self.dataset = dataset

    def compute(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.

        Returns:
            Dictionary with various statistics
        """
        if not self.dataset.examples:
            return {
                "total_examples": 0,
                "warning": "Empty dataset"
            }

        # Basic counts
        total_examples = len(self.dataset.examples)
        examples_with_gt = sum(
            1 for ex in self.dataset.examples
            if ex.ground_truth_answer
        )
        examples_with_scores = sum(
            1 for ex in self.dataset.examples
            if ex.relevance_scores
        )

        # Relevant documents statistics
        relevant_doc_counts = [
            len(ex.relevant_doc_ids)
            for ex in self.dataset.examples
        ]
        avg_relevant_docs = sum(relevant_doc_counts) / len(relevant_doc_counts)
        min_relevant_docs = min(relevant_doc_counts)
        max_relevant_docs = max(relevant_doc_counts)

        # Query length statistics
        query_lengths = [
            len(ex.query.split())
            for ex in self.dataset.examples
        ]
        avg_query_length = sum(query_lengths) / len(query_lengths)
        min_query_length = min(query_lengths)
        max_query_length = max(query_lengths)

        # Answer length statistics (if available)
        answer_lengths = [
            len(ex.ground_truth_answer.split())
            for ex in self.dataset.examples
            if ex.ground_truth_answer
        ]
        if answer_lengths:
            avg_answer_length = sum(answer_lengths) / len(answer_lengths)
            min_answer_length = min(answer_lengths)
            max_answer_length = max(answer_lengths)
        else:
            avg_answer_length = None
            min_answer_length = None
            max_answer_length = None

        # Relevance score distribution (if available)
        all_relevance_scores = []
        for ex in self.dataset.examples:
            all_relevance_scores.extend(ex.relevance_scores.values())

        if all_relevance_scores:
            score_distribution = Counter(all_relevance_scores)
            avg_relevance_score = sum(all_relevance_scores) / len(all_relevance_scores)
        else:
            score_distribution = {}
            avg_relevance_score = None

        return {
            "dataset_name": self.dataset.name,
            "total_examples": total_examples,
            "examples_with_ground_truth": examples_with_gt,
            "examples_with_relevance_scores": examples_with_scores,
            "relevant_documents": {
                "average": round(avg_relevant_docs, 2),
                "min": min_relevant_docs,
                "max": max_relevant_docs,
                "distribution": Counter(relevant_doc_counts)
            },
            "query_length": {
                "average_words": round(avg_query_length, 2),
                "min_words": min_query_length,
                "max_words": max_query_length
            },
            "answer_length": {
                "average_words": round(avg_answer_length, 2) if avg_answer_length else None,
                "min_words": min_answer_length,
                "max_words": max_answer_length
            } if answer_lengths else None,
            "relevance_scores": {
                "average": round(avg_relevance_score, 2) if avg_relevance_score else None,
                "distribution": dict(score_distribution)
            } if all_relevance_scores else None,
            "metadata": self.dataset.metadata
        }

    def format_summary(self, stats: Dict[str, Any]) -> str:
        """
        Format statistics as human-readable string.

        Args:
            stats: Statistics dictionary from compute()

        Returns:
            Formatted string
        """
        lines = [
            f"Dataset: {stats['dataset_name']}",
            f"Total Examples: {stats['total_examples']}",
            f"Examples with Ground Truth: {stats['examples_with_ground_truth']}",
            f"Examples with Relevance Scores: {stats['examples_with_relevance_scores']}",
            "",
            "Relevant Documents:",
            f"  Average: {stats['relevant_documents']['average']}",
            f"  Min: {stats['relevant_documents']['min']}",
            f"  Max: {stats['relevant_documents']['max']}",
            "",
            "Query Length:",
            f"  Average: {stats['query_length']['average_words']} words",
            f"  Min: {stats['query_length']['min_words']} words",
            f"  Max: {stats['query_length']['max_words']} words",
        ]

        if stats.get('answer_length'):
            lines.extend([
                "",
                "Answer Length:",
                f"  Average: {stats['answer_length']['average_words']} words",
                f"  Min: {stats['answer_length']['min_words']} words",
                f"  Max: {stats['answer_length']['max_words']} words",
            ])

        if stats.get('relevance_scores'):
            lines.extend([
                "",
                "Relevance Scores:",
                f"  Average: {stats['relevance_scores']['average']}",
                f"  Distribution: {stats['relevance_scores']['distribution']}",
            ])

        return '\n'.join(lines)

    def get_quality_issues(self) -> List[str]:
        """
        Identify potential quality issues in the dataset.

        Returns:
            List of warning messages about quality issues
        """
        issues = []

        if not self.dataset.examples:
            issues.append("Dataset is empty")
            return issues

        # Check for examples without relevant documents
        no_relevant = sum(
            1 for ex in self.dataset.examples
            if not ex.relevant_doc_ids
        )
        if no_relevant > 0:
            pct = (no_relevant / len(self.dataset.examples)) * 100
            issues.append(
                f"{no_relevant} examples ({pct:.1f}%) have no relevant documents"
            )

        # Check for duplicate query IDs
        query_ids = [ex.query_id for ex in self.dataset.examples]
        duplicates = [qid for qid, count in Counter(query_ids).items() if count > 1]
        if duplicates:
            issues.append(f"Found {len(duplicates)} duplicate query IDs")

        # Check for very short queries
        short_queries = sum(
            1 for ex in self.dataset.examples
            if len(ex.query.split()) < 3
        )
        if short_queries > 0:
            pct = (short_queries / len(self.dataset.examples)) * 100
            issues.append(
                f"{short_queries} queries ({pct:.1f}%) are very short (< 3 words)"
            )

        # Check for missing ground truth when expected
        no_gt_pct = (
            (len(self.dataset.examples) - sum(1 for ex in self.dataset.examples if ex.ground_truth_answer))
            / len(self.dataset.examples)
        ) * 100
        if no_gt_pct > 50:
            issues.append(
                f"{no_gt_pct:.1f}% of examples are missing ground truth answers"
            )

        return issues
