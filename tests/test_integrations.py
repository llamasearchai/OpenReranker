"""Test cases for OpenReranker integrations."""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Test DSPy integration if available
try:
    import dspy

    from open_reranker.integrations.dspy.reranker import DSPyReranker

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Test LangChain integration if available
try:
    from langchain_core.documents import Document as LangChainDocument

    from open_reranker.integrations.langchain.reranker import LangChainReranker

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
class TestDSPyReranker:
    """Test cases for DSPy integration."""

    @patch("open_reranker.integrations.dspy.reranker.RerankerService")
    def test_init(self, mock_service):
        """Test DSPyReranker initialization."""
        mock_base_retriever = Mock()

        reranker = DSPyReranker(
            base_retriever=mock_base_retriever,
            model="test-model",
            top_k=5,
            use_mlx=False,
        )

        assert reranker.base_retriever == mock_base_retriever
        assert reranker.model == "test-model"
        assert reranker.top_k == 5
        assert reranker.use_mlx is False
        mock_service.assert_called_once()

    @patch("open_reranker.integrations.dspy.reranker.RerankerService")
    def test_forward_with_string_query(self, mock_service):
        """Test forward method with string query."""
        mock_service_instance = Mock()
        mock_service_instance.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service.return_value = mock_service_instance

        mock_base_retriever = Mock()
        mock_base_retriever.forward.return_value = [("doc1", 0.5), ("doc2", 0.4)]

        reranker = DSPyReranker(base_retriever=mock_base_retriever)

        result = reranker.forward("test query", k=2)

        assert len(result) == 2
        assert result[0] == ("doc1", 0.9)
        assert result[1] == ("doc2", 0.8)
        mock_service_instance.rerank.assert_called_once()

    @patch("open_reranker.integrations.dspy.reranker.RerankerService")
    def test_forward_with_prediction_object(self, mock_service):
        """Test forward method with prediction object."""
        mock_service_instance = Mock()
        mock_service_instance.rerank.return_value = [(0, 0.9)]
        mock_service.return_value = mock_service_instance

        mock_base_retriever = Mock()
        mock_base_retriever.forward.return_value = [("doc1", 0.5)]

        mock_prediction = Mock()
        mock_prediction.question = "test question"

        reranker = DSPyReranker(base_retriever=mock_base_retriever)

        result = reranker.forward(mock_prediction)

        assert len(result) == 1
        assert result[0] == ("doc1", 0.9)
        mock_service_instance.rerank.assert_called_with(
            query="test question",
            documents=["doc1"],
            model="default",
            top_k=10,
            include_scores=True,
            use_mlx=True,
        )

    @patch("open_reranker.integrations.dspy.reranker.RerankerService")
    def test_forward_no_base_retriever(self, mock_service):
        """Test forward method without base retriever."""
        mock_service_instance = Mock()
        mock_service.return_value = mock_service_instance

        reranker = DSPyReranker(base_retriever=None)

        result = reranker.forward("test query")

        assert result == []
        mock_service_instance.rerank.assert_not_called()

    @patch("open_reranker.integrations.dspy.reranker.RerankerService")
    def test_retrieve(self, mock_service):
        """Test retrieve method."""
        mock_service_instance = Mock()
        mock_service_instance.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service.return_value = mock_service_instance

        mock_base_retriever = Mock()
        mock_base_retriever.forward.return_value = [("doc1", 0.5), ("doc2", 0.4)]

        reranker = DSPyReranker(base_retriever=mock_base_retriever)

        result = reranker.retrieve("test query", k=2)

        assert len(result) == 2
        assert result[0] == "doc1"
        assert result[1] == "doc2"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestLangChainReranker:
    """Test cases for LangChain integration."""

    @patch("open_reranker.integrations.langchain.reranker.RerankerService")
    def test_init(self, mock_service):
        """Test LangChainReranker initialization."""
        from langchain_core.retrievers import BaseRetriever

        # Create a proper mock that looks like BaseRetriever
        mock_base_retriever = MagicMock(spec=BaseRetriever)

        reranker = LangChainReranker(
            base_retriever=mock_base_retriever,
            model="test-model",
            top_k=5,
            use_mlx=False,
        )

        assert reranker.base_retriever == mock_base_retriever
        assert reranker.model == "test-model"
        assert reranker.top_k == 5
        assert reranker.use_mlx is False
        mock_service.assert_called_once()

    @patch("open_reranker.integrations.langchain.reranker.RerankerService")
    def test_get_relevant_documents(self, mock_service):
        """Test getting relevant documents."""
        from langchain_core.retrievers import BaseRetriever

        mock_service_instance = Mock()
        mock_service_instance.rerank.return_value = [(0, 0.9), (1, 0.8)]
        mock_service.return_value = mock_service_instance

        # Create mock base documents
        base_docs = [
            LangChainDocument(page_content="doc1", metadata={"source": "test1"}),
            LangChainDocument(page_content="doc2", metadata={"source": "test2"}),
        ]

        mock_base_retriever = MagicMock(spec=BaseRetriever)
        mock_base_retriever.get_relevant_documents.return_value = base_docs

        mock_run_manager = Mock()

        reranker = LangChainReranker(base_retriever=mock_base_retriever)

        result = reranker._get_relevant_documents(
            "test query", run_manager=mock_run_manager
        )

        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[0].metadata["reranker_score"] == 0.9
        assert result[1].page_content == "doc2"
        assert result[1].metadata["reranker_score"] == 0.8

        mock_service_instance.rerank.assert_called_once_with(
            query="test query",
            documents=["doc1", "doc2"],
            model="default",
            top_k=10,
            include_scores=True,
            use_mlx=True,
        )

    @patch("open_reranker.integrations.langchain.reranker.RerankerService")
    def test_get_relevant_documents_no_base_retriever(self, mock_service):
        """Test getting relevant documents without base retriever."""
        mock_service_instance = Mock()
        mock_service.return_value = mock_service_instance

        mock_run_manager = Mock()

        reranker = LangChainReranker(base_retriever=None)

        result = reranker._get_relevant_documents(
            "test query", run_manager=mock_run_manager
        )

        assert result == []
        mock_service_instance.rerank.assert_not_called()

    @patch("open_reranker.integrations.langchain.reranker.RerankerService")
    def test_get_relevant_documents_empty_base_results(self, mock_service):
        """Test getting relevant documents with empty base results."""
        from langchain_core.retrievers import BaseRetriever

        mock_service_instance = Mock()
        mock_service.return_value = mock_service_instance

        mock_base_retriever = MagicMock(spec=BaseRetriever)
        mock_base_retriever.get_relevant_documents.return_value = []

        mock_run_manager = Mock()

        reranker = LangChainReranker(base_retriever=mock_base_retriever)

        result = reranker._get_relevant_documents(
            "test query", run_manager=mock_run_manager
        )

        assert result == []
        mock_service_instance.rerank.assert_not_called()

    @patch("open_reranker.integrations.langchain.reranker.RerankerService")
    def test_metadata_preservation(self, mock_service):
        """Test that original metadata is preserved and reranker score is added."""
        from langchain_core.retrievers import BaseRetriever

        mock_service_instance = Mock()
        mock_service_instance.rerank.return_value = [(0, 0.95)]
        mock_service.return_value = mock_service_instance

        # Create mock base document with existing metadata
        original_metadata = {"source": "test.txt", "page": 1, "author": "test"}
        base_docs = [
            LangChainDocument(page_content="test content", metadata=original_metadata)
        ]

        mock_base_retriever = MagicMock(spec=BaseRetriever)
        mock_base_retriever.get_relevant_documents.return_value = base_docs

        mock_run_manager = Mock()

        reranker = LangChainReranker(base_retriever=mock_base_retriever)

        result = reranker._get_relevant_documents(
            "test query", run_manager=mock_run_manager
        )

        assert len(result) == 1
        assert result[0].page_content == "test content"

        # Check that original metadata is preserved
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["page"] == 1
        assert result[0].metadata["author"] == "test"

        # Check that reranker score is added
        assert result[0].metadata["reranker_score"] == 0.95
