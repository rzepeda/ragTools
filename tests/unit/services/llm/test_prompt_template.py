"""Unit tests for prompt template system."""

import pytest
from rag_factory.services.llm.prompt_template import PromptTemplate, CommonTemplates
from rag_factory.services.llm.base import MessageRole


def test_template_with_system_and_user():
    """Test template with system and user messages."""
    template = PromptTemplate(system="You are a {role}", user="Hello {name}")

    messages = template.format(role="assistant", name="Alice")

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == "You are a assistant"
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Hello Alice"


def test_template_user_only():
    """Test template with user message only."""
    template = PromptTemplate(user="Question: {question}")

    messages = template.format(question="What is AI?")

    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Question: What is AI?"


def test_template_with_few_shot_examples():
    """Test template with few-shot examples."""
    template = PromptTemplate(
        system="You are a classifier",
        few_shot_examples=[
            {"user": "Text: Happy day", "assistant": "Positive"},
            {"user": "Text: Sad day", "assistant": "Negative"},
        ],
        user="Text: {text}",
    )

    messages = template.format(text="Great weather")

    assert len(messages) == 6  # system + 2 examples (2 msgs each) + user
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[5].content == "Text: Great weather"


def test_template_missing_variable_raises_error():
    """Test template with missing variable raises error."""
    template = PromptTemplate(user="Hello {name}")

    with pytest.raises(ValueError, match="Missing template variable"):
        template.format()


def test_template_validation():
    """Test template validation."""
    template = PromptTemplate(user="Hello {name}")

    assert template.validate(name="Alice") is True
    assert template.validate() is False


def test_common_template_rag_qa():
    """Test RAG QA common template."""
    messages = CommonTemplates.RAG_QA.format(
        context="The sky is blue.", question="What color is the sky?"
    )

    assert len(messages) == 2
    assert "context" in messages[1].content.lower()
    assert "question" in messages[1].content.lower()


def test_common_template_summarization():
    """Test summarization common template."""
    messages = CommonTemplates.SUMMARIZATION.format(
        text="Long text here...", max_words="50"
    )

    assert len(messages) == 2
    assert "summarize" in messages[1].content.lower()


def test_template_multiple_variables():
    """Test template with multiple variables."""
    template = PromptTemplate(user="Name: {name}, Age: {age}, City: {city}")

    messages = template.format(name="Alice", age=30, city="NYC")

    assert messages[0].content == "Name: Alice, Age: 30, City: NYC"


def test_template_with_assistant_prefix():
    """Test template with assistant message prefix."""
    template = PromptTemplate(
        user="Complete this sentence: The weather is",
        assistant="currently",
    )

    messages = template.format()

    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == "currently"


def test_common_template_classification():
    """Test classification common template."""
    messages = CommonTemplates.CLASSIFICATION.format(
        text="I love this product!", categories="Positive, Negative, Neutral"
    )

    assert len(messages) == 2
    assert "classify" in messages[1].content.lower()
    assert "Positive" in messages[1].content


def test_common_template_extraction():
    """Test extraction common template."""
    messages = CommonTemplates.EXTRACTION.format(
        text="John Doe lives in NYC", fields="name, location"
    )

    assert len(messages) == 2
    assert "extract" in messages[1].content.lower()
    assert "JSON" in messages[1].content


def test_empty_template():
    """Test empty template."""
    template = PromptTemplate()

    messages = template.format()

    assert len(messages) == 0


def test_template_with_special_characters():
    """Test template with special characters."""
    template = PromptTemplate(user="Process: {text}")

    messages = template.format(text="Hello\nWorld\t!")

    assert "Hello\nWorld\t!" in messages[0].content
