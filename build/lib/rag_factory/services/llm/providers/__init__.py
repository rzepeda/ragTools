"""LLM provider implementations."""

# Lazy imports to avoid requiring all dependencies

__all__ = ["AnthropicProvider", "OpenAIProvider", "OllamaProvider"]


def __getattr__(name):
    """Lazy import providers to avoid requiring all dependencies."""
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    elif name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    elif name == "OllamaProvider":
        from .ollama import OllamaProvider

        return OllamaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
