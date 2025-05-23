"""Test cases for OpenReranker services."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from open_reranker.models.cross_encoder import CrossEncoder
from open_reranker.services.reranker_service import RerankerService


class TestRerankerService:
    """Test cases for RerankerService."""

    def test_init(self):
        """Test RerankerService initialization."""
        service = RerankerService()
        assert service.text_rerankers == {}
        assert service.code_reranker is None
        assert service.table_reranker is None
        assert service.accelerator is None
        assert service.last_rerank_time == 0.0

    @patch("open_reranker.services.reranker_service.TextReranker")
    def test_get_text_reranker_default(self, mock_text_reranker):
        """Test getting default text reranker."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_text_reranker.return_value = mock_reranker

        result = service._get_text_reranker("default")

        assert result == mock_reranker
        mock_text_reranker.assert_called_once()
        assert "jina/reranker-v2" in service.text_rerankers

    @patch("open_reranker.services.reranker_service.TextReranker")
    def test_get_text_reranker_cached(self, mock_text_reranker):
        """Test getting cached text reranker."""
        service = RerankerService()
        mock_reranker = Mock()
        service.text_rerankers["test-model"] = mock_reranker

        result = service._get_text_reranker("test-model")

        assert result == mock_reranker
        mock_text_reranker.assert_not_called()

    @patch("open_reranker.services.reranker_service.CodeReranker")
    def test_get_code_reranker(self, mock_code_reranker):
        """Test getting code reranker."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_code_reranker.return_value = mock_reranker

        result = service._get_code_reranker()

        assert result == mock_reranker
        assert service.code_reranker == mock_reranker
        mock_code_reranker.assert_called_once()

    @patch("open_reranker.services.reranker_service.TableReranker")
    def test_get_table_reranker(self, mock_table_reranker):
        """Test getting table reranker."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_table_reranker.return_value = mock_reranker

        result = service._get_table_reranker()

        assert result == mock_reranker
        assert service.table_reranker == mock_reranker
        mock_table_reranker.assert_called_once()

    @patch("open_reranker.services.reranker_service.TextReranker")
    def test_rerank_basic(self, mock_text_reranker):
        """Test basic reranking functionality."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_reranker.compute_scores.return_value = np.array([0.9, 0.8, 0.7])
        mock_text_reranker.return_value = mock_reranker

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        result = service.rerank(query, documents)

        assert len(result) == 3
        assert result[0] == (0, 0.9)
        assert result[1] == (1, 0.8)
        assert result[2] == (2, 0.7)
        assert service.last_rerank_time > 0

    @patch("open_reranker.services.reranker_service.TextReranker")
    def test_rerank_with_top_k(self, mock_text_reranker):
        """Test reranking with top_k parameter."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_reranker.compute_scores.return_value = np.array([0.9, 0.8, 0.7])
        mock_text_reranker.return_value = mock_reranker

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        result = service.rerank(query, documents, top_k=2)

        assert len(result) == 2
        assert result[0] == (0, 0.9)
        assert result[1] == (1, 0.8)

    def test_rerank_empty_documents(self):
        """Test reranking with empty documents list."""
        service = RerankerService()

        result = service.rerank("test query", [])

        assert result == []

    @patch("open_reranker.services.reranker_service.TextReranker")
    @patch("open_reranker.services.reranker_service.truncate_text")
    def test_rerank_with_truncation(self, mock_truncate, mock_text_reranker):
        """Test reranking with text truncation."""
        service = RerankerService()
        mock_reranker = Mock()
        mock_reranker.compute_scores.return_value = np.array([0.9])
        mock_text_reranker.return_value = mock_reranker
        mock_truncate.side_effect = lambda text, max_len: text[:max_len]

        long_query = "x" * 2000
        long_doc = "y" * 10000

        result = service.rerank(long_query, [long_doc])

        assert len(result) == 1
        mock_truncate.assert_called()


class TestCrossEncoder:
    """Test cases for CrossEncoder."""

    @patch("open_reranker.services.cross_encoder.AutoTokenizer")
    @patch("open_reranker.services.cross_encoder.AutoModelForSequenceClassification")
    @patch("torch.cuda.is_available")
    def test_init(self, mock_cuda, mock_model, mock_tokenizer):
        """Test CrossEncoder initialization."""
        mock_cuda.return_value = False
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        encoder = CrossEncoder("test-model")

        assert encoder.model_name == "test-model"
        assert encoder.tokenizer == mock_tokenizer_instance
        assert encoder.model == mock_model_instance
        assert encoder.device.type == "cpu"

    @patch("open_reranker.services.cross_encoder.AutoTokenizer")
    @patch("open_reranker.services.cross_encoder.AutoModelForSequenceClassification")
    @patch("torch.cuda.is_available")
    def test_init_with_gpu(self, mock_cuda, mock_model, mock_tokenizer):
        """Test CrossEncoder initialization with GPU."""
        mock_cuda.return_value = True
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        encoder = CrossEncoder("test-model")

        assert encoder.device.type == "cuda"

    @patch("open_reranker.services.cross_encoder.AutoTokenizer")
    @patch("open_reranker.services.cross_encoder.AutoModelForSequenceClassification")
    @patch("torch.cuda.is_available")
    def test_compute_scores(self, mock_cuda, mock_model, mock_tokenizer):
        """Test computing scores."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock tokenizer output
        mock_features = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_tokenizer_instance.return_value = mock_features

        # Mock model output
        mock_output = Mock()
        mock_logits = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output

        encoder = CrossEncoder("test-model")

        query = "test query"
        documents = ["doc1", "doc2"]

        scores = encoder.compute_scores(query, documents)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        np.testing.assert_array_almost_equal(scores, [0.9, 0.8])

    @patch("open_reranker.services.cross_encoder.AutoTokenizer")
    @patch("open_reranker.services.cross_encoder.AutoModelForSequenceClassification")
    @patch("torch.cuda.is_available")
    def test_compute_scores_single_output(self, mock_cuda, mock_model, mock_tokenizer):
        """Test computing scores with single output model."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock tokenizer output
        mock_features = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer_instance.return_value = mock_features

        # Mock model output with single dimension
        mock_output = Mock()
        mock_logits = torch.tensor([0.9])
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output

        encoder = CrossEncoder("test-model")

        query = "test query"
        documents = ["doc1"]

        scores = encoder.compute_scores(query, documents)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 1
        np.testing.assert_array_almost_equal(scores, [0.9])

    def test_set_accelerator(self):
        """Test setting accelerator."""
        with (
            patch("open_reranker.services.cross_encoder.AutoTokenizer"),
            patch(
                "open_reranker.services.cross_encoder.AutoModelForSequenceClassification"
            ),
            patch("torch.cuda.is_available", return_value=False),
        ):

            encoder = CrossEncoder("test-model")
            mock_accelerator = Mock()

            encoder.set_accelerator(mock_accelerator)

            assert encoder.accelerator == mock_accelerator
