"""Test LM Studio connection and model availability."""

import os
import pytest
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("LM_STUDIO_BASE_URL"),
    reason="LM_STUDIO_BASE_URL not set"
)
def test_lm_studio_models_endpoint():
    """Test that LM Studio is running and has models loaded."""
    base_url = os.getenv("LM_STUDIO_BASE_URL")
    expected_model = os.getenv("LM_STUDIO_MODEL")
    
    # Remove /v1 suffix if present to get base URL
    if base_url.endswith("/v1"):
        models_url = f"{base_url}/models"
    else:
        models_url = f"{base_url}/v1/models"
    
    try:
        response = requests.get(models_url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        # Check response structure
        assert "data" in data, "Response should have 'data' field"
        assert isinstance(data["data"], list), "'data' should be a list"
        
        # Get list of model IDs
        model_ids = [model.get("id") for model in data["data"]]
        
        print(f"\n✅ LM Studio is running at {base_url}")
        print(f"📋 Available models: {model_ids}")
        print(f"🎯 Expected model: {expected_model}")
        
        # Check if expected model is loaded
        assert len(model_ids) > 0, "No models loaded in LM Studio"
        assert expected_model in model_ids, (
            f"Model '{expected_model}' not found in LM Studio. "
            f"Available models: {model_ids}"
        )
        
        print(f"✅ Model '{expected_model}' is loaded and ready!")
        
    except requests.exceptions.ConnectionError:
        pytest.fail(
            f"❌ Cannot connect to LM Studio at {base_url}. "
            "Make sure LM Studio is running on the host machine."
        )
    except requests.exceptions.Timeout:
        pytest.fail(
            f"❌ Connection to LM Studio at {base_url} timed out. "
            "Check network connectivity."
        )
    except Exception as e:
        pytest.fail(f"❌ Error checking LM Studio: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("LM_STUDIO_BASE_URL"),
    reason="LM_STUDIO_BASE_URL not set"
)
def test_lm_studio_completion():
    """Test that LM Studio can generate completions."""
    from rag_factory.services.llm import LLMService, LLMServiceConfig, Message, MessageRole
    
    config = LLMServiceConfig(
        provider="openai",
        model=os.getenv("LM_STUDIO_MODEL", "local-model"),
        provider_config={
            "api_key": os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            "base_url": os.getenv("LM_STUDIO_BASE_URL")
        },
        enable_rate_limiting=False,
    )
    
    service = LLMService(config)
    
    messages = [Message(role=MessageRole.USER, content="Say 'Hello' and nothing else.")]
    
    try:
        response = service.complete(messages, max_tokens=10)
        
        assert response.content, "Response should have content"
        assert response.total_tokens > 0, "Should have token count"
        assert response.cost == 0.0, "Local model should have zero cost"
        assert response.provider == "openai", "Should use openai provider"
        
        print(f"\n✅ LM Studio completion successful!")
        print(f"📝 Response: {response.content}")
        print(f"🔢 Tokens: {response.total_tokens}")
        
    except Exception as e:
        pytest.fail(f"❌ LM Studio completion failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    print("Testing LM Studio connection...\n")
    test_lm_studio_models_endpoint()
    print("\nTesting LM Studio completion...\n")
    test_lm_studio_completion()
    print("\n✅ All LM Studio tests passed!")
