#!/usr/bin/env python3
"""Simple test to verify LLM connection and response parsing."""

import os
from dotenv import load_dotenv
from rag_factory.services.llm.service import LLMService
from rag_factory.services.llm.config import LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole

# Load environment variables
load_dotenv()

# Create LLM service using LM Studio config from .env
config = LLMServiceConfig(
    provider="openai",
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    provider_config={
        "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
        "base_url": os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    }
)

llm_service = LLMService(config)

# Simple test: expand "machine learning" with keywords
messages = [
    Message(
        role=MessageRole.SYSTEM,
        content="You are a helpful assistant that expands search queries with relevant keywords."
    ),
    Message(
        role=MessageRole.USER,
        content='Original query: "machine learning"\n\nExpand this query by adding 5 relevant keywords or synonyms.\nReturn only the expanded query, nothing else.\n\nExpanded query:'
    )
]

print("Calling LLM...")
response = llm_service.complete(messages=messages, temperature=0.7, max_tokens=150)

print("\n=== LLM Response ===")
print(f"Content: '{response.content}'")
print(f"Content length: {len(response.content)}")
print(f"Content (repr): {repr(response.content)}")
print(f"\nPrompt tokens: {response.prompt_tokens}")
print(f"Completion tokens: {response.completion_tokens}")
print(f"Total tokens: {response.total_tokens}")
print(f"Cost: ${response.cost}")
print(f"Latency: {response.latency:.2f}s")

# Test stripping
stripped = response.content.strip()
print(f"\n=== After .strip() ===")
print(f"Stripped: '{stripped}'")
print(f"Stripped length: {len(stripped)}")

# Test quote stripping
if stripped.startswith('"') and stripped.endswith('"'):
    unquoted = stripped[1:-1].strip()
    print(f"\n=== After removing quotes ===")
    print(f"Unquoted: '{unquoted}'")
    print(f"Unquoted length: {len(unquoted)}")
else:
    print("\n=== No quotes to remove ===")
