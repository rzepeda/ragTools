"""Unit tests for CustomModelLoader."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from rag_factory.models.embedding.loader import CustomModelLoader
from rag_factory.models.embedding.models import ModelConfig, ModelFormat, PoolingStrategy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


@pytest.fixture
def onnx_config():
    """Sample ONNX config."""
    return ModelConfig(
        model_path="model.onnx",
        model_format=ModelFormat.ONNX,
        device="cpu",
        tokenizer_name="cl100k_base"
    )


def test_loader_initialization(loader):
    """Test loader initializes with empty cache."""
    assert isinstance(loader.loaded_models, dict)
    assert len(loader.loaded_models) == 0


def test_model_caching(loader, sentence_transformer_config):
    """Test that models are cached after first load."""
    mock_st_module = MagicMock()
    mock_st_class = MagicMock()
    mock_st_module.SentenceTransformer = mock_st_class
    
    with patch.dict('sys.modules', {'sentence_transformers': mock_st_module}):
        mock_model = Mock()
        mock_st_class.return_value = mock_model
        
        # First load
        model1 = loader.load_model(sentence_transformer_config)
        
        # Second load should use cache
        model2 = loader.load_model(sentence_transformer_config)
        
        # Should be same object
        assert model1 is model2
        # SentenceTransformer should only be called once
        assert mock_st_class.call_count == 1


def test_embed_texts_sentence_transformer(loader, sentence_transformer_config):
    """Test embedding with Sentence-Transformers."""
    mock_st_module = MagicMock()
    mock_st_class = MagicMock()
    mock_st_module.SentenceTransformer = mock_st_class
    
    # Patch sys.modules to ensure import works, AND patch the class if it was already imported (unlikely but safe)
    with patch.dict('sys.modules', {'sentence_transformers': mock_st_module}):
        # Mock model
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_st_class.return_value = mock_model
        
        # Load model
        model = loader.load_model(sentence_transformer_config)
        
        # Generate embeddings
        texts = ["Hello", "World"]
        embeddings = loader.embed_texts(texts, model, sentence_transformer_config)
        
        # Verify
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        mock_model.encode.assert_called_once()


def test_embed_onnx(loader, onnx_config):
    """Test ONNX embedding generation."""
    mock_ort_module = MagicMock()
    mock_session_class = MagicMock()
    mock_ort_module.InferenceSession = mock_session_class
    mock_ort_module.get_available_providers.return_value = ['CPUExecutionProvider']
    mock_ort_module.SessionOptions.return_value = Mock()
    mock_ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL = 1
    
    mock_tiktoken_module = MagicMock()
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2]
    mock_tiktoken_module.get_encoding.return_value = mock_encoding
    
    # Patch module-level variables in loader AND sys.modules
    with patch('rag_factory.models.embedding.loader.tiktoken', mock_tiktoken_module), \
         patch('rag_factory.models.embedding.loader.TIKTOKEN_AVAILABLE', True), \
         patch('rag_factory.models.embedding.loader.NUMPY_AVAILABLE', True), \
         patch('rag_factory.models.embedding.loader.np', np), \
         patch.dict('sys.modules', {
             'onnxruntime': mock_ort_module,
             'tiktoken': mock_tiktoken_module
         }):
        
        # Mock ONNX session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock session run output (last_hidden_state)
        # Batch=1, Seq=2, Dim=4
        mock_output = [np.ones((1, 2, 4))]
        mock_session.run.return_value = mock_output
        
        # Load model
        model = loader.load_model(onnx_config)
        
        # Generate embeddings
        texts = ["test"]
        embeddings = loader.embed_texts(texts, model, onnx_config)
        
        # Verify
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 4
        mock_session.run.assert_called_once()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for HF tests")
def test_huggingface_embedding_batching(loader, huggingface_config):
    """Test that Hugging Face embedding processes in batches."""
    mock_transformers_module = MagicMock()
    mock_model_class = MagicMock()
    mock_tokenizer_class = MagicMock()
    mock_transformers_module.AutoModel = mock_model_class
    mock_transformers_module.AutoTokenizer = mock_tokenizer_class
    
    with patch.dict('sys.modules', {'transformers': mock_transformers_module}):
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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for pooling tests")
def test_pool_embeddings_mean(loader):
    """Test mean pooling strategy (PyTorch)."""
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


def test_pool_embeddings_numpy_mean(loader):
    """Test mean pooling strategy (NumPy)."""
    token_embeddings = np.array([
        [[1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0],
         [3.0, 4.0, 5.0, 6.0]],
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [0.0, 0.0, 0.0, 0.0]]
    ])
    
    attention_mask = np.array([
        [1, 1, 1],
        [1, 1, 0]
    ])
    
    pooled = loader._pool_embeddings_numpy(
        token_embeddings,
        attention_mask,
        PoolingStrategy.MEAN
    )
    
    assert pooled.shape == (2, 4)
    expected_first = np.array([2.0, 3.0, 4.0, 5.0])
    assert np.allclose(pooled[0], expected_first)
    
    expected_second = np.array([1.5, 1.5, 1.5, 1.5])
    assert np.allclose(pooled[1], expected_second)


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


def test_embed_onnx(loader, onnx_config):
    """Test ONNX embedding generation."""
    mock_ort_module = MagicMock()
    mock_session_class = MagicMock()
    mock_ort_module.InferenceSession = mock_session_class
    mock_ort_module.get_available_providers.return_value = ['CPUExecutionProvider']
    mock_ort_module.SessionOptions.return_value = Mock()
    mock_ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL = 1
    
    mock_tiktoken_module = MagicMock()
    
    with patch.dict('sys.modules', {
        'onnxruntime': mock_ort_module,
        'tiktoken': mock_tiktoken_module,
        'numpy': np
    }):
        # Mock ONNX session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock session run output (last_hidden_state)
        # Batch=1, Seq=2, Dim=4
        mock_output = [np.ones((1, 2, 4))]
        mock_session.run.return_value = mock_output
        
        # Mock tokenizer
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2]
        mock_tiktoken_module.get_encoding.return_value = mock_encoding
        
        # Load model
        model = loader.load_model(onnx_config)
        
        # Generate embeddings
        texts = ["test"]
        embeddings = loader.embed_texts(texts, model, onnx_config)
        
        # Verify
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 4
        mock_session.run.assert_called_once()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for HF tests")
def test_huggingface_embedding_batching(loader, huggingface_config):
    """Test that Hugging Face embedding processes in batches."""
    mock_transformers_module = MagicMock()
    mock_model_class = MagicMock()
    mock_tokenizer_class = MagicMock()
    mock_transformers_module.AutoModel = mock_model_class
    mock_transformers_module.AutoTokenizer = mock_tokenizer_class
    
    with patch.dict('sys.modules', {'transformers': mock_transformers_module}):
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
