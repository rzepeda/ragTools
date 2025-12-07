"""Unit tests for CustomModelLoader."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from rag_factory.models.embedding.loader import CustomModelLoader
from rag_factory.models.embedding.models import ModelConfig, ModelFormat, PoolingStrategy


@pytest.fixture
def loader():
    """Create CustomModelLoader instance."""
    return CustomModelLoader()


@pytest.fixture
def sentence_transformer_config():
    """Sample Sentence-Transformers config."""
    return ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )


@pytest.fixture
def huggingface_config():
    """Sample Hugging Face config."""
    return ModelConfig(
        model_path="bert-base-uncased",
        model_format=ModelFormat.HUGGINGFACE,
        device="cpu",
        pooling_strategy=PoolingStrategy.MEAN
    )


def test_loader_initialization(loader):
    """Test loader initializes with empty cache."""
    assert isinstance(loader.loaded_models, dict)
    assert len(loader.loaded_models) == 0


def test_model_caching(loader, sentence_transformer_config):
    """Test that models are cached after first load."""
    with patch('rag_factory.models.embedding.loader.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        # First load
        model1 = loader.load_model(sentence_transformer_config)
        
        # Second load should use cache
        model2 = loader.load_model(sentence_transformer_config)
        
        # Should be same object
        assert model1 is model2
        # SentenceTransformer should only be called once
        assert mock_st.call_count == 1


def test_load_sentence_transformer_missing_dependency(loader, sentence_transformer_config):
    """Test error when sentence-transformers not installed."""
    with patch.dict('sys.modules', {'sentence_transformers': None}):
        with pytest.raises(ImportError, match="sentence-transformers is required"):
            loader.load_model(sentence_transformer_config)


def test_load_huggingface_missing_dependency(loader, huggingface_config):
    """Test error when transformers not installed."""
    with patch.dict('sys.modules', {'transformers': None}):
        with pytest.raises(ImportError, match="transformers is required"):
            loader.load_model(huggingface_config)


def test_unsupported_format(loader):
    """Test error for unsupported model format."""
    config = ModelConfig(
        model_path="some/path",
        model_format=ModelFormat.PYTORCH,  # Not implemented
        device="cpu"
    )
    
    with pytest.raises(ValueError, match="Unsupported model format"):
        loader.load_model(config)


def test_pool_embeddings_mean(loader):
    """Test mean pooling strategy."""
    # Create sample token embeddings [batch=2, seq_len=3, hidden_dim=4]
    token_embeddings = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0],
         [3.0, 4.0, 5.0, 6.0]],
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [0.0, 0.0, 0.0, 0.0]]  # This will be masked
    ])
    
    # Attention mask [batch=2, seq_len=3]
    attention_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]  # Last token masked
    ])
    
    pooled = loader._pool_embeddings(
        token_embeddings,
        attention_mask,
        PoolingStrategy.MEAN
    )
    
    # Check shape
    assert pooled.shape == (2, 4)
    
    # Check first batch (mean of all 3 tokens)
    expected_first = torch.tensor([2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(pooled[0], expected_first)
    
    # Check second batch (mean of first 2 tokens only)
    expected_second = torch.tensor([1.5, 1.5, 1.5, 1.5])
    assert torch.allclose(pooled[1], expected_second)


def test_pool_embeddings_cls(loader):
    """Test CLS pooling strategy."""
    token_embeddings = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0],
         [3.0, 4.0, 5.0, 6.0]]
    ])
    
    attention_mask = torch.tensor([[1, 1, 1]])
    
    pooled = loader._pool_embeddings(
        token_embeddings,
        attention_mask,
        PoolingStrategy.CLS
    )
    
    # Should return first token
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(pooled, expected)


def test_pool_embeddings_max(loader):
    """Test max pooling strategy."""
    token_embeddings = torch.tensor([
        [[1.0, 5.0, 2.0, 3.0],
         [2.0, 3.0, 6.0, 4.0],
         [4.0, 2.0, 3.0, 7.0]]
    ])
    
    attention_mask = torch.tensor([[1, 1, 1]])
    
    pooled = loader._pool_embeddings(
        token_embeddings,
        attention_mask,
        PoolingStrategy.MAX
    )
    
    # Should return max across sequence dimension
    expected = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
    assert torch.allclose(pooled, expected)


def test_pool_embeddings_unknown_strategy(loader):
    """Test error for unknown pooling strategy."""
    token_embeddings = torch.randn(1, 3, 4)
    attention_mask = torch.ones(1, 3)
    
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        loader._pool_embeddings(
            token_embeddings,
            attention_mask,
            PoolingStrategy.WEIGHTED_MEAN  # Not implemented
        )


def test_embed_texts_sentence_transformer(loader, sentence_transformer_config):
    """Test embedding with Sentence-Transformers."""
    with patch('rag_factory.models.embedding.loader.SentenceTransformer') as mock_st:
        # Mock model
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        # Load model
        model = loader.load_model(sentence_transformer_config)
        
        # Generate embeddings
        texts = ["Hello", "World"]
        embeddings = loader.embed_texts(texts, model, sentence_transformer_config)
        
        # Verify
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        mock_model.encode.assert_called_once()


def test_embed_texts_unsupported_format(loader):
    """Test error for unsupported format in embed_texts."""
    config = ModelConfig(
        model_path="some/path",
        model_format=ModelFormat.PYTORCH,
        device="cpu"
    )
    
    with pytest.raises(ValueError, match="Unsupported model format"):
        loader.embed_texts(["test"], None, config)


def test_embed_onnx_not_implemented(loader):
    """Test that ONNX embedding raises NotImplementedError."""
    config = ModelConfig(
        model_path="model.onnx",
        model_format=ModelFormat.ONNX,
        device="cpu"
    )
    
    with patch('rag_factory.models.embedding.loader.ort') as mock_ort:
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1
        
        model = loader.load_model(config)
        
        with pytest.raises(NotImplementedError, match="ONNX embedding not yet implemented"):
            loader.embed_texts(["test"], model, config)


def test_huggingface_embedding_batching(loader, huggingface_config):
    """Test that Hugging Face embedding processes in batches."""
    with patch('rag_factory.models.embedding.loader.AutoModel') as mock_model_class, \
         patch('rag_factory.models.embedding.loader.AutoTokenizer') as mock_tokenizer_class:
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 5, 768)  # batch=2, seq=5, hidden=768
        mock_model.return_value = mock_output
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 5)),
            'attention_mask': torch.ones(2, 5)
        }
        
        # Load model
        model_dict = loader.load_model(huggingface_config)
        
        # Generate embeddings with small batch size
        huggingface_config.batch_size = 2
        texts = ["text1", "text2", "text3", "text4"]
        
        embeddings = loader.embed_texts(texts, model_dict, huggingface_config)
        
        # Should process in 2 batches
        assert len(embeddings) == 4
        assert mock_tokenizer.call_count == 2  # 2 batches
