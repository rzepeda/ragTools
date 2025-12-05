"""Prompt template system for LLM prompts."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from .base import Message, MessageRole


@dataclass
class PromptTemplate:
    """Template for LLM prompts with variable substitution.

    Example:
        template = PromptTemplate(
            system="You are a helpful assistant.",
            user="Answer this question: {question}\nContext: {context}"
        )

        messages = template.format(
            question="What is AI?",
            context="AI stands for Artificial Intelligence..."
        )

    Attributes:
        system: System message template
        user: User message template
        assistant: Assistant message template (for prefilling)
        few_shot_examples: List of few-shot examples
    """

    system: Optional[str] = None
    user: Optional[str] = None
    assistant: Optional[str] = None
    few_shot_examples: Optional[List[Dict[str, str]]] = field(default_factory=list)

    def format(self, **variables) -> List[Message]:
        """Format template with variables.

        Args:
            **variables: Template variables to substitute

        Returns:
            List of formatted messages

        Raises:
            ValueError: If required template variable is missing
        """
        messages = []

        # Add system message
        if self.system:
            content = self._substitute(self.system, variables)
            messages.append(Message(role=MessageRole.SYSTEM, content=content))

        # Add few-shot examples
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=self._substitute(example.get("user", ""), variables),
                    )
                )
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=self._substitute(
                            example.get("assistant", ""), variables
                        ),
                    )
                )

        # Add user message
        if self.user:
            content = self._substitute(self.user, variables)
            messages.append(Message(role=MessageRole.USER, content=content))

        # Add assistant prefix (if any)
        if self.assistant:
            content = self._substitute(self.assistant, variables)
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))

        return messages

    def _substitute(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in template.

        Args:
            template: Template string
            variables: Variables to substitute

        Returns:
            Formatted string

        Raises:
            ValueError: If required variable is missing
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}") from e

    def validate(self, **variables) -> bool:
        """Validate that all required variables are provided.

        Args:
            **variables: Variables to validate

        Returns:
            True if all variables are valid
        """
        try:
            self.format(**variables)
            return True
        except ValueError:
            return False


class CommonTemplates:
    """Collection of common prompt templates."""

    RAG_QA = PromptTemplate(
        system="You are a helpful assistant that answers questions based on provided context.",
        user="""Answer the following question based on the context provided.

Context:
{context}

Question: {question}

Answer:""",
    )

    SUMMARIZATION = PromptTemplate(
        system="You are an expert at summarizing text concisely.",
        user="""Summarize the following text in {max_words} words or less:

{text}

Summary:""",
    )

    CLASSIFICATION = PromptTemplate(
        system="You are a text classifier.",
        user="""Classify the following text into one of these categories: {categories}

Text: {text}

Category:""",
    )

    EXTRACTION = PromptTemplate(
        system="You extract structured information from text.",
        user="""Extract the following information from the text: {fields}

Text: {text}

Extracted information (JSON format):""",
    )

    REWRITE = PromptTemplate(
        system="You are an expert at rewriting text to improve clarity and style.",
        user="""Rewrite the following text to {goal}:

{text}

Rewritten text:""",
    )

    TRANSLATION = PromptTemplate(
        system="You are a professional translator.",
        user="""Translate the following text from {source_language} to {target_language}:

{text}

Translation:""",
    )
