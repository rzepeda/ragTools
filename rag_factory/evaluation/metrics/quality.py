"""
Quality evaluation metrics for generated answers.

This module provides metrics for evaluating the quality of generated answers,
including semantic similarity, faithfulness to context, and relevance.
"""

from typing import List, Optional
import numpy as np
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticSimilarity(IMetric):
    """
    Semantic similarity between generated answer and ground truth.

    Uses sentence embeddings to compute cosine similarity between the
    generated answer and ground truth answer. Requires sentence-transformers.

    Args:
        model_name: Name of sentence transformer model (default: "all-MiniLM-L6-v2")

    Example:
        >>> metric = SemanticSimilarity()
        >>> result = metric.compute(
        ...     generated_answer="Paris is the capital of France",
        ...     ground_truth="The capital of France is Paris"
        ... )
        >>> print(f"Similarity: {result.value}")  # High similarity ~0.9+
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Semantic Similarity metric.

        Args:
            model_name: Sentence transformer model name

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SemanticSimilarity. "
                "Install it with: pip install sentence-transformers"
            )
        super().__init__("semantic_similarity", MetricType.QUALITY)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def compute(
        self,
        generated_answer: str,
        ground_truth: str,
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute semantic similarity.

        Args:
            generated_answer: Generated answer text
            ground_truth: Ground truth answer text
            query_id: Optional query identifier

        Returns:
            MetricResult with similarity score (0.0 to 1.0)
        """
        if not generated_answer or not ground_truth:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"warning": "Empty input text"},
                query_id=query_id
            )

        # Generate embeddings
        embeddings = self.model.encode([generated_answer, ground_truth])

        # Compute cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return MetricResult(
            name=self.name,
            value=float(similarity),
            metadata={
                "model": self.model_name,
                "embedding_dim": self.model.get_sentence_embedding_dimension()
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Semantic similarity between generated and ground truth answers using embeddings"


class Faithfulness(IMetric):
    """
    Faithfulness: Whether answer is grounded in retrieved context.

    Measures if claims in the answer can be verified from the context.
    This is a simplified implementation using word overlap. For production,
    consider using NLI models or LLM-based verification.

    Example:
        >>> metric = Faithfulness()
        >>> result = metric.compute(
        ...     answer="Paris is the capital of France",
        ...     context=["France is a country in Europe. Paris is its capital city."]
        ... )
        >>> print(f"Faithfulness: {result.value}")
    """

    def __init__(self, overlap_threshold: float = 0.5):
        """
        Initialize Faithfulness metric.

        Args:
            overlap_threshold: Minimum word overlap ratio to consider grounded (default: 0.5)
        """
        super().__init__("faithfulness", MetricType.QUALITY)
        self.overlap_threshold = overlap_threshold

    def compute(
        self,
        answer: str,
        context: List[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute faithfulness score.

        Args:
            answer: Generated answer
            context: Retrieved context documents
            query_id: Optional query identifier

        Returns:
            MetricResult with faithfulness score (0.0 to 1.0)
        """
        if not answer or not context:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"warning": "Empty answer or context"},
                query_id=query_id
            )

        # Convert to lowercase for comparison
        answer_lower = answer.lower()
        context_text = " ".join(context).lower()

        # Split answer into sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if not sentences:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"warning": "No sentences in answer"},
                query_id=query_id
            )

        grounded_count = 0
        for sentence in sentences:
            # Check if sentence words appear in context
            words = sentence.split()
            if not words:
                continue

            word_coverage = sum(1 for word in words if word in context_text)
            if word_coverage / len(words) >= self.overlap_threshold:
                grounded_count += 1

        faithfulness_score = grounded_count / len(sentences)

        return MetricResult(
            name=self.name,
            value=faithfulness_score,
            metadata={
                "total_sentences": len(sentences),
                "grounded_sentences": grounded_count,
                "overlap_threshold": self.overlap_threshold
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Degree to which answer is grounded in retrieved context"


class AnswerRelevance(IMetric):
    """
    Answer Relevance: How relevant the answer is to the query.

    Uses semantic similarity between query and answer to determine if the
    answer addresses the question. Requires sentence-transformers.

    Args:
        model_name: Name of sentence transformer model (default: "all-MiniLM-L6-v2")

    Example:
        >>> metric = AnswerRelevance()
        >>> result = metric.compute(
        ...     query="What is the capital of France?",
        ...     answer="Paris is the capital of France"
        ... )
        >>> print(f"Relevance: {result.value}")  # High relevance
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Answer Relevance metric.

        Args:
            model_name: Sentence transformer model name

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for AnswerRelevance. "
                "Install it with: pip install sentence-transformers"
            )
        super().__init__("answer_relevance", MetricType.QUALITY)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def compute(
        self,
        query: str,
        answer: str,
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute answer relevance.

        Args:
            query: User query
            answer: Generated answer
            query_id: Optional query identifier

        Returns:
            MetricResult with relevance score (0.0 to 1.0)
        """
        if not query or not answer:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"warning": "Empty query or answer"},
                query_id=query_id
            )

        # Generate embeddings
        embeddings = self.model.encode([query, answer])

        # Compute cosine similarity
        relevance = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return MetricResult(
            name=self.name,
            value=float(relevance),
            metadata={
                "model": self.model_name,
                "embedding_dim": self.model.get_sentence_embedding_dimension()
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Semantic similarity between query and answer"


class AnswerCompleteness(IMetric):
    """
    Answer Completeness: Coverage of key information from ground truth.

    Measures what fraction of key terms/concepts from the ground truth
    are present in the generated answer.

    Example:
        >>> metric = AnswerCompleteness()
        >>> result = metric.compute(
        ...     generated_answer="Paris is the capital",
        ...     ground_truth="Paris is the capital of France with 2 million people"
        ... )
        >>> print(f"Completeness: {result.value}")
    """

    def __init__(self, min_word_length: int = 3):
        """
        Initialize Answer Completeness metric.

        Args:
            min_word_length: Minimum word length to consider as key term (default: 3)
        """
        super().__init__("answer_completeness", MetricType.QUALITY)
        self.min_word_length = min_word_length

    def compute(
        self,
        generated_answer: str,
        ground_truth: str,
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute answer completeness.

        Args:
            generated_answer: Generated answer text
            ground_truth: Ground truth answer text
            query_id: Optional query identifier

        Returns:
            MetricResult with completeness score (0.0 to 1.0)
        """
        if not generated_answer or not ground_truth:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"warning": "Empty input"},
                query_id=query_id
            )

        # Extract key terms from ground truth (simple: longer words)
        ground_truth_lower = ground_truth.lower()
        generated_lower = generated_answer.lower()

        # Get significant words from ground truth
        key_terms = set(
            word.strip('.,!?;:')
            for word in ground_truth_lower.split()
            if len(word.strip('.,!?;:')) >= self.min_word_length
        )

        if not key_terms:
            return MetricResult(
                name=self.name,
                value=1.0,  # No key terms to match
                metadata={"warning": "No key terms found"},
                query_id=query_id
            )

        # Count how many key terms appear in generated answer
        covered_terms = sum(1 for term in key_terms if term in generated_lower)
        completeness = covered_terms / len(key_terms)

        return MetricResult(
            name=self.name,
            value=completeness,
            metadata={
                "total_key_terms": len(key_terms),
                "covered_terms": covered_terms,
                "min_word_length": self.min_word_length
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Coverage of key information from ground truth in generated answer"
