"""Pytest configuration and fixtures for OpenReranker tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging
from open_reranker.models.cross_encoder import CrossEncoder
from open_reranker.main import app
from open_reranker.services.reranker_service import RerankerService


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_reranker_service():
    """Create a mock reranker service."""
    service = Mock(spec=RerankerService)
    service.last_rerank_time = 0.1
    service.rerank.return_value = [(0, 0.9), (1, 0.8), (2, 0.7)]
    return service


@pytest.fixture
def mock_cross_encoder():
    """Create a mock cross encoder."""
    encoder = Mock(spec=CrossEncoder)
    encoder.compute_scores.return_value = np.array([0.9, 0.8, 0.7])
    encoder.model_name = "test-model"
    encoder.device = "cpu"
    return encoder


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "doc1", "text": "This is about machine learning and AI."},
        {"id": "doc2", "text": "Python programming language tutorial."},
        {"id": "doc3", "text": "The weather is nice today."},
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "machine learning tutorial"


@pytest.fixture
def sample_code_documents():
    """Sample code documents for testing."""
    return [
        'def hello_world():\n    print("Hello, World!")',
        "import numpy as np\ndef calculate_mean(data):\n    return np.mean(data)",
        "for i in range(10):\n    print(i)",
    ]


@pytest.fixture
def sample_table_data():
    """Sample table data for testing."""
    return {
        "headers": ["Name", "Age", "City"],
        "rows": [
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
            ["Charlie", "35", "Chicago"],
        ],
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Override settings for testing
    with patch.object(settings, "DEBUG", True):
        with patch.object(settings, "USE_MLX", False):
            yield


@pytest.fixture
def mock_transformers():
    """Mock transformers components."""
    with (
        patch("open_reranker.services.cross_encoder.AutoTokenizer") as mock_tokenizer,
        patch(
            "open_reranker.services.cross_encoder.AutoModelForSequenceClassification"
        ) as mock_model,
    ):

        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure mock model
        mock_model_instance = Mock()
        mock_output = Mock()
        mock_output.logits = Mock()
        mock_output.logits.shape = [3, 2]  # batch_size=3, num_classes=2
        mock_output.logits.__getitem__.return_value = np.array([0.9, 0.8, 0.7])
        mock_output.logits.cpu.return_value.numpy.return_value = np.array(
            [0.9, 0.8, 0.7]
        )
        mock_model_instance.return_value = mock_output
        mock_model.from_pretrained.return_value = mock_model_instance

        yield mock_tokenizer, mock_model
