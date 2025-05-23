"""Test cases for the OpenReranker API."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from open_reranker.api.models import Document, RerankerRequest
from open_reranker.core.auth import AuthManager
from open_reranker.core.config import settings


# Helper to create a valid token for testing
def create_test_token(user_id: str = "test_user", tier: str = "free"):
    auth_manager = AuthManager()
    return auth_manager.create_access_token(data={"sub": user_id, "tier": tier})


class TestAPI:
    """Test cases for the OpenReranker API endpoints."""

    def test_health_check(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "open-reranker"
        assert data["version"] == "1.0.0"

    def test_metrics_endpoint(self, test_client):
        """Test the metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert isinstance(response.text, str)

    @patch("open_reranker.api.router.RerankerService")
    def test_rerank_endpoint_success(
        self, mock_service_class, test_client, sample_documents, sample_query
    ):
        """Test successful reranking."""
        # Setup mock
        mock_service = Mock()
        mock_service.rerank.return_value = [(0, 0.9), (1, 0.8), (2, 0.7)]
        mock_service.last_rerank_time = 0.1
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": sample_query,
            "documents": sample_documents,
            "model": "default",
            "top_k": 3,
            "include_scores": True,
        }

        # Make request
        response = test_client.post("/api/v1/rerank", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == sample_query
        assert len(data["results"]) == 3
        assert data["results"][0]["score"] == 0.9
        assert data["timing"]["reranking_time"] == 0.1

    @patch("open_reranker.api.router.RerankerService")
    def test_rerank_endpoint_with_top_k(
        self, mock_service_class, test_client, sample_documents, sample_query
    ):
        """Test reranking with top_k parameter."""
        # Setup mock
        mock_service = Mock()
        mock_service.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service.last_rerank_time = 0.1
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": sample_query,
            "documents": sample_documents,
            "model": "default",
            "top_k": 2,
            "include_scores": True,
        }

        # Make request
        response = test_client.post("/api/v1/rerank", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    @patch("open_reranker.api.router.RerankerService")
    def test_rerank_endpoint_without_scores(
        self, mock_service_class, test_client, sample_documents, sample_query
    ):
        """Test reranking without scores."""
        # Setup mock
        mock_service = Mock()
        mock_service.rerank.return_value = [(0, 0.9), (1, 0.8), (2, 0.7)]
        mock_service.last_rerank_time = 0.1
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": sample_query,
            "documents": sample_documents,
            "model": "default",
            "include_scores": False,
        }

        # Make request
        response = test_client.post("/api/v1/rerank", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["score"] is None

    def test_rerank_endpoint_invalid_request(self, test_client):
        """Test reranking with invalid request data."""
        # Missing required fields
        request_data = {
            "query": "test query"
            # Missing documents
        }

        response = test_client.post("/api/v1/rerank", json=request_data)
        assert response.status_code == 422  # Validation error

    @patch("open_reranker.api.router.RerankerService")
    def test_rerank_endpoint_service_error(
        self, mock_service_class, test_client, sample_documents, sample_query
    ):
        """Test reranking when service throws an error."""
        # Setup mock to raise exception
        mock_service = Mock()
        mock_service.rerank.side_effect = Exception("Service error")
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": sample_query,
            "documents": sample_documents,
            "model": "default",
        }

        # Make request
        response = test_client.post("/api/v1/rerank", json=request_data)

        # Assertions
        assert response.status_code == 500

    @patch("open_reranker.api.router.RerankerService")
    def test_dspy_integration_endpoint(self, mock_service_class, test_client):
        """Test DSPy integration endpoint."""
        # Setup mock
        mock_service = Mock()
        mock_service.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "model": "default",
            "top_k": 2,
        }

        # Make request
        response = test_client.post("/api/v1/integrations/dspy", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 2
        assert data["results"][0] == ["doc1", 0.9]

    @patch("open_reranker.api.router.RerankerService")
    def test_langchain_integration_endpoint(self, mock_service_class, test_client):
        """Test LangChain integration endpoint."""
        # Setup mock
        mock_service = Mock()
        mock_service.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service_class.return_value = mock_service

        # Prepare request
        request_data = {
            "query": "test query",
            "documents": [
                {"page_content": "doc1", "metadata": {}},
                {"page_content": "doc2", "metadata": {}},
            ],
            "model": "default",
            "top_k": 2,
        }

        # Make request
        response = test_client.post("/api/v1/integrations/langchain", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 2
        assert data["results"][0]["metadata"]["reranker_score"] == 0.9

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        # Test with a valid GET request to health endpoint
        response = test_client.get("/")
        # CORS headers should be present for cross-origin requests
        # For same-origin requests, they might not be added
        # To properly test CORS, we simulate a cross-origin request by adding an Origin header
        headers = {"Origin": "http://test-cors.com"}
        response = test_client.get("/", headers=headers)
        assert response.status_code == 200  # Should be 200 if CORS is handled
        assert "access-control-allow-origin" in response.headers
        assert (
            response.headers["access-control-allow-origin"] == "*"
        )  # Or specific origin

    def test_process_time_header(self, test_client):
        """Test that process time header is added."""
        response = test_client.get("/")
        assert "x-process-time" in response.headers
        assert float(response.headers["x-process-time"]) >= 0


class TestAPIModels:
    """Test cases for API models."""

    def test_document_model(self):
        """Test Document model."""
        doc = Document(id="test-id", text="test text", metadata={"key": "value"})
        assert doc.id == "test-id"
        assert doc.text == "test text"
        assert doc.metadata == {"key": "value"}

    def test_reranker_request_model(self, sample_documents):
        """Test RerankerRequest model."""
        documents = [Document(**doc) for doc in sample_documents]
        request = RerankerRequest(
            query="test query", documents=documents, model="test-model", top_k=5
        )
        assert request.query == "test query"
        assert len(request.documents) == 3
        assert request.model == "test-model"
        assert request.top_k == 5
        assert request.include_scores is True  # default value

    def test_reranker_request_defaults(self, sample_documents):
        """Test RerankerRequest model defaults."""
        req = RerankerRequest(query="test", documents=sample_documents)
        assert req.model == settings.DEFAULT_RERANKER_MODEL
        assert (
            req.top_k is None
        )  # Default in Pydantic model, API might apply another default
        assert req.include_scores is True
        assert req.use_mlx == settings.USE_MLX


# --- Tests for Auth, Rate Limiting, Cache --- #


@patch("open_reranker.api.router.RerankerService")
@patch("open_reranker.core.auth.settings.AUTH_ENABLED", True)
@patch(
    "open_reranker.core.rate_limiting.settings.RATE_LIMIT_QPS", 1
)  # Stricter limit for test
async def test_rerank_auth_and_rate_limit(
    mock_service_class, test_client, sample_documents, sample_query
):
    """Test auth and rate limiting on /rerank endpoint."""
    mock_service = Mock()
    mock_service.rerank = Mock(return_value=[(0, 0.9)])
    mock_service.last_rerank_time = 0.1
    mock_service_class.return_value = mock_service

    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    request_data = {"query": sample_query, "documents": sample_documents}

    # First call - should pass
    response = test_client.post("/api/v1/rerank", json=request_data, headers=headers)
    assert response.status_code == 200

    # Second call immediately - should be rate limited (QPS=1)
    response = test_client.post("/api/v1/rerank", json=request_data, headers=headers)
    assert response.status_code == 429

    # Call without token - should fail if auth is on (it's patched to True)
    response = test_client.post("/api/v1/rerank", json=request_data)
    assert response.status_code == 401  # Unauthorized


@patch("open_reranker.api.router.RerankerService")
@patch("open_reranker.core.cache.cache_manager.get_rerank_result")
@patch("open_reranker.core.cache.cache_manager.set_rerank_result")
async def test_rerank_caching(
    mock_set_cache,
    mock_get_cache,
    mock_service_class,
    test_client,
    sample_documents,
    sample_query,
):
    """Test caching for /rerank endpoint."""
    mock_service = Mock()
    mock_service.rerank = Mock(
        return_value=[(0, 0.9)]
    )  # This should only be called once if cache works
    mock_service.last_rerank_time = 0.1
    mock_service_class.return_value = mock_service

    request_data = {
        "query": sample_query,
        "documents": sample_documents,
        "model": "cached_model",
    }

    # First call: cache miss
    mock_get_cache.return_value = None
    response1 = test_client.post("/api/v1/rerank", json=request_data)
    assert response1.status_code == 200
    mock_get_cache.assert_called_once()
    mock_service.rerank.assert_called_once()
    mock_set_cache.assert_called_once()
    assert response1.json()["metadata"]["cached"] is False

    # Second call: cache hit
    mock_get_cache.reset_mock()
    mock_set_cache.reset_mock()
    mock_service.rerank.reset_mock()
    # Simulate cache returning the previously computed result
    # The structure should match what RerankerService.rerank returns: List[Tuple[int, float]]
    cached_data = [(0, 0.9)]
    mock_get_cache.return_value = cached_data

    response2 = test_client.post("/api/v1/rerank", json=request_data)
    assert response2.status_code == 200
    mock_get_cache.assert_called_once()
    mock_service.rerank.assert_not_called()  # Should not call service again
    mock_set_cache.assert_not_called()  # Should not set cache again
    assert response2.json()["metadata"]["cached"] is True
    assert response2.json()["results"][0]["score"] == 0.9


@patch("open_reranker.api.router.RerankerService")
@patch("open_reranker.core.auth.settings.AUTH_ENABLED", True)
async def test_batch_rerank_auth_and_tier(
    mock_service_class, test_client, sample_documents, sample_query
):
    """Test auth and tier check for /rerank/batch endpoint."""
    mock_service = Mock()
    mock_service.rerank = Mock(return_value=[(0, 0.9)])
    mock_service.last_rerank_time = 0.1
    mock_service_class.return_value = mock_service

    batch_request_data = {
        "queries": [sample_query, "another query"],
        "documents": [sample_documents, sample_documents],
        "model": "default",
    }

    # No token
    response = test_client.post("/api/v1/rerank/batch", json=batch_request_data)
    assert response.status_code == 401

    # Free tier token - should fail (default required tier is "pro")
    free_token = create_test_token(tier="free")
    headers_free = {"Authorization": f"Bearer {free_token}"}
    response = test_client.post(
        "/api/v1/rerank/batch", json=batch_request_data, headers=headers_free
    )
    assert response.status_code == 403  # Forbidden due to tier

    # Pro tier token - should pass
    pro_token = create_test_token(tier="pro")
    headers_pro = {"Authorization": f"Bearer {pro_token}"}
    response = test_client.post(
        "/api/v1/rerank/batch", json=batch_request_data, headers=headers_pro
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


@patch("open_reranker.core.cache.cache_manager.clear")
@patch("open_reranker.core.auth.settings.AUTH_ENABLED", True)
async def test_clear_cache_endpoint(mock_clear_cache, test_client):
    """Test /cache DELETE endpoint for auth and tier."""
    mock_clear_cache.return_value = None  # Simulate successful clear

    # No token
    response = test_client.delete("/api/v1/cache")
    assert response.status_code == 401

    # Pro tier token (requires "enterprise")
    pro_token = create_test_token(tier="pro")
    headers_pro = {"Authorization": f"Bearer {pro_token}"}
    response = test_client.delete("/api/v1/cache", headers=headers_pro)
    assert response.status_code == 403

    # Enterprise tier token
    enterprise_token = create_test_token(tier="enterprise")
    headers_enterprise = {"Authorization": f"Bearer {enterprise_token}"}
    response = test_client.delete("/api/v1/cache", headers=headers_enterprise)
    assert response.status_code == 200
    assert response.json() == {"message": "Cache cleared successfully"}
    mock_clear_cache.assert_called_once()
