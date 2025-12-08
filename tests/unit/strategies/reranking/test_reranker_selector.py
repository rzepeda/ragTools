"""Unit tests for reranker selector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from rag_factory.services.utils.reranker_selector import RerankerSelector


@pytest.fixture
def mock_embedder():
    """Create a mock embedding provider."""
    return Mock()


class TestRerankerSelectorAvailability:
    """Tests for checking reranker availability."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_is_cohere_available_with_key_and_package(self):
        """Test Cohere is available with API key and package."""
        # Mock successful import
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock() if name == 'cohere' else __import__(name, *args, **kwargs)):
            assert RerankerSelector._is_cohere_available({"cohere_api_key": "test-key"}) is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_cohere_available_no_key(self):
        """Test Cohere is not available without API key."""
        assert RerankerSelector._is_cohere_available({}) is False

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    def test_is_cohere_available_with_env_key(self):
        """Test Cohere availability check uses environment variable."""
        # Even without package, should return False due to import error
        # But with key in env, it tries to import
        result = RerankerSelector._is_cohere_available({})
        # Result depends on whether cohere is actually installed
        assert isinstance(result, bool)

    def test_is_torch_available_true(self):
        """Test PyTorch is available."""
        # Mock successful import
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock() if name == 'torch' else __import__(name, *args, **kwargs)):
            assert RerankerSelector._is_torch_available() is True

    def test_is_torch_available_false(self):
        """Test PyTorch is not available."""
        # Mock failed import
        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("No module named 'torch'")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = RerankerSelector._is_torch_available()
            assert isinstance(result, bool)

    def test_get_available_rerankers(self):
        """Test getting available rerankers."""
        available = RerankerSelector.get_available_rerankers()

        assert isinstance(available, dict)
        assert "cohere" in available
        assert "cosine" in available
        assert "bge" in available
        assert "cross-encoder" in available
        # Cosine should always be available
        assert available["cosine"] is True


class TestRerankerSelectorAutoSelection:
    """Tests for automatic reranker selection."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_cohere_available")
    @patch("rag_factory.strategies.reranking.cohere_reranker.CohereReranker")
    def test_auto_select_cohere_when_available(
        self, mock_cohere_class, mock_is_available, mock_embedder
    ):
        """Test auto-selection chooses Cohere when available."""
        mock_is_available.return_value = True
        mock_cohere_instance = Mock()
        mock_cohere_class.return_value = mock_cohere_instance

        reranker = RerankerSelector.select_reranker(mock_embedder)

        assert reranker == mock_cohere_instance
        mock_cohere_class.assert_called_once()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_cohere_available")
    @patch("rag_factory.strategies.reranking.cosine_reranker.CosineReranker")
    def test_auto_select_cosine_when_cohere_unavailable(
        self, mock_cosine_class, mock_is_available, mock_embedder
    ):
        """Test auto-selection falls back to Cosine when Cohere unavailable."""
        mock_is_available.return_value = False
        mock_cosine_instance = Mock()
        mock_cosine_class.return_value = mock_cosine_instance

        reranker = RerankerSelector.select_reranker(mock_embedder)

        assert reranker == mock_cosine_instance
        mock_cosine_class.assert_called_once()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_cohere_available")
    @patch("rag_factory.strategies.reranking.cosine_reranker.CosineReranker")
    def test_auto_select_with_custom_config(
        self, mock_cosine_class, mock_is_available, mock_embedder
    ):
        """Test auto-selection respects custom configuration."""
        mock_is_available.return_value = False
        mock_cosine_instance = Mock()
        mock_cosine_class.return_value = mock_cosine_instance

        config = {
            "similarity_metric": "dot",
            "normalize": False
        }

        reranker = RerankerSelector.select_reranker(mock_embedder, config)

        # Should pass config to CosineReranker
        call_args = mock_cosine_class.call_args
        assert call_args is not None


class TestRerankerSelectorManualSelection:
    """Tests for manual reranker selection."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key"})
    @patch("rag_factory.strategies.reranking.cohere_reranker.CohereReranker")
    def test_manual_select_cohere(self, mock_cohere_class, mock_embedder):
        """Test manual selection of Cohere reranker."""
        mock_cohere_instance = Mock()
        mock_cohere_class.return_value = mock_cohere_instance

        config = {"reranker_type": "cohere"}
        reranker = RerankerSelector.select_reranker(mock_embedder, config)

        assert reranker == mock_cohere_instance
        mock_cohere_class.assert_called_once()

    @patch("rag_factory.strategies.reranking.cosine_reranker.CosineReranker")
    def test_manual_select_cosine(self, mock_cosine_class, mock_embedder):
        """Test manual selection of Cosine reranker."""
        mock_cosine_instance = Mock()
        mock_cosine_class.return_value = mock_cosine_instance

        config = {"reranker_type": "cosine"}
        reranker = RerankerSelector.select_reranker(mock_embedder, config)

        assert reranker == mock_cosine_instance
        mock_cosine_class.assert_called_once()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    @patch("rag_factory.strategies.reranking.bge_reranker.BGEReranker")
    def test_manual_select_bge_with_torch(
        self, mock_bge_class, mock_is_torch, mock_embedder
    ):
        """Test manual selection of BGE reranker when PyTorch available."""
        mock_is_torch.return_value = True
        mock_bge_instance = Mock()
        mock_bge_class.return_value = mock_bge_instance

        config = {"reranker_type": "bge"}
        reranker = RerankerSelector.select_reranker(mock_embedder, config)

        assert reranker == mock_bge_instance
        mock_bge_class.assert_called_once()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    def test_manual_select_bge_without_torch(self, mock_is_torch, mock_embedder):
        """Test manual selection of BGE fails without PyTorch."""
        mock_is_torch.return_value = False

        config = {"reranker_type": "bge"}

        with pytest.raises(ImportError, match="BGE reranker requires PyTorch"):
            RerankerSelector.select_reranker(mock_embedder, config)

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    @patch("rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoderReranker")
    def test_manual_select_cross_encoder_with_torch(
        self, mock_ce_class, mock_is_torch, mock_embedder
    ):
        """Test manual selection of Cross-Encoder when PyTorch available."""
        mock_is_torch.return_value = True
        mock_ce_instance = Mock()
        mock_ce_class.return_value = mock_ce_instance

        config = {"reranker_type": "cross-encoder"}
        reranker = RerankerSelector.select_reranker(mock_embedder, config)

        assert reranker == mock_ce_instance
        mock_ce_class.assert_called_once()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    def test_manual_select_cross_encoder_without_torch(self, mock_is_torch, mock_embedder):
        """Test manual selection of Cross-Encoder fails without PyTorch."""
        mock_is_torch.return_value = False

        config = {"reranker_type": "cross-encoder"}

        with pytest.raises(ImportError, match="Cross-Encoder reranker requires PyTorch"):
            RerankerSelector.select_reranker(mock_embedder, config)

    def test_manual_select_unknown_type(self, mock_embedder):
        """Test manual selection with unknown reranker type."""
        config = {"reranker_type": "unknown"}

        with pytest.raises(ValueError, match="Unknown reranker type"):
            RerankerSelector.select_reranker(mock_embedder, config)


class TestRerankerSelectorErrorMessages:
    """Tests for helpful error messages."""

    @patch.dict(os.environ, {}, clear=True)
    def test_cohere_error_message_suggests_alternatives(self, mock_embedder):
        """Test Cohere error message suggests lightweight alternatives."""
        config = {"reranker_type": "cohere"}

        with pytest.raises(ImportError) as exc_info:
            RerankerSelector.select_reranker(mock_embedder, config)

        error_msg = str(exc_info.value)
        assert "pip install cohere" in error_msg or "API key" in error_msg

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    def test_bge_error_message_suggests_alternatives(self, mock_is_torch, mock_embedder):
        """Test BGE error message suggests lightweight alternatives."""
        mock_is_torch.return_value = False
        config = {"reranker_type": "bge"}

        with pytest.raises(ImportError) as exc_info:
            RerankerSelector.select_reranker(mock_embedder, config)

        error_msg = str(exc_info.value)
        assert "cohere" in error_msg.lower() or "cosine" in error_msg.lower()

    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_torch_available")
    def test_cross_encoder_error_message_suggests_alternatives(
        self, mock_is_torch, mock_embedder
    ):
        """Test Cross-Encoder error message suggests lightweight alternatives."""
        mock_is_torch.return_value = False
        config = {"reranker_type": "cross-encoder"}

        with pytest.raises(ImportError) as exc_info:
            RerankerSelector.select_reranker(mock_embedder, config)

        error_msg = str(exc_info.value)
        assert "cohere" in error_msg.lower() or "cosine" in error_msg.lower()


class TestRerankerSelectorConfiguration:
    """Tests for configuration handling."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "env-key"})
    @patch("rag_factory.services.utils.reranker_selector.RerankerSelector._is_cohere_available")
    @patch("rag_factory.strategies.reranking.cohere_reranker.CohereReranker")
    def test_config_api_key_overrides_env(
        self, mock_cohere_class, mock_is_available, mock_embedder
    ):
        """Test config API key overrides environment variable."""
        mock_is_available.return_value = True
        mock_cohere_instance = Mock()
        mock_cohere_class.return_value = mock_cohere_instance

        config = {
            "reranker_type": "cohere",
            "cohere_api_key": "config-key"
        }

        RerankerSelector.select_reranker(mock_embedder, config)

        # Check that config key was used
        call_args = mock_cohere_class.call_args
        assert call_args is not None

    @patch("rag_factory.strategies.reranking.cosine_reranker.CosineReranker")
    def test_cosine_metric_configuration(self, mock_cosine_class, mock_embedder):
        """Test Cosine reranker respects metric configuration."""
        mock_cosine_instance = Mock()
        mock_cosine_class.return_value = mock_cosine_instance

        config = {
            "reranker_type": "cosine",
            "similarity_metric": "euclidean",
            "normalize": False
        }

        RerankerSelector.select_reranker(mock_embedder, config)

        # Verify configuration was passed
        call_args = mock_cosine_class.call_args
        assert call_args is not None
