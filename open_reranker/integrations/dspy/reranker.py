import asyncio
from typing import Any, List, Optional, Tuple

from dspy.retrieve.retriever import Retriever

from open_reranker.core.logging import setup_logging
from open_reranker.services.reranker_service import RerankerService

logger = setup_logging()


class DSPyReranker(Retriever):
    """
    DSPy-compatible reranker that wraps OpenReranker service.

    This class allows using OpenReranker as a retriever in DSPy pipelines.
    It can be used on top of another retriever to rerank the results.
    """

    def __init__(
        self,
        base_retriever: Optional[Retriever] = None,
        model: str = "default",
        top_k: int = 10,
        use_mlx: bool = True,
    ):
        """
        Initialize the DSPy Reranker.

        Args:
            base_retriever: Optional base retriever to wrap
            model: Model name to use for reranking
            top_k: Number of top results to keep
            use_mlx: Whether to use MLX acceleration if available
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.reranker_service = RerankerService()
        self.model = model
        self.top_k = top_k
        self.use_mlx = use_mlx

        logger.info(f"Initialized DSPyReranker with model={model}, top_k={top_k}")

    async def forward(
        self, query_or_pred: str, k: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve and rerank passages for the given query.

        Args:
            query_or_pred: Query string or DSPy prediction
            k: Number of results to return (overrides self.top_k)

        Returns:
            List of (text, score) tuples
        """
        # Get the query text from various possible input types
        if isinstance(query_or_pred, str):
            query = query_or_pred
        elif hasattr(query_or_pred, "question"):
            query = query_or_pred.question
        elif hasattr(query_or_pred, "query"):
            query = query_or_pred.query
        else:
            try:
                # Try to convert to string
                query = str(query_or_pred)
            except:
                raise ValueError(f"Cannot extract query from input: {query_or_pred}")

        # Get top_k to use
        top_k = k if k is not None else self.top_k

        # If we have a base retriever, use it to get initial results
        if self.base_retriever:
            base_results = self.base_retriever.forward(query, k=min(100, top_k * 3))

            # Extract documents from base results
            documents = []
            document_metadata = []
            for text, score in base_results:
                documents.append(text)
                document_metadata.append({"base_score": score})
        else:
            # No base retriever, nothing to rerank
            logger.warning("No base retriever and no documents provided")
            return []

        # If we have no documents, return empty results
        if not documents:
            return []

        # Rerank the documents
        reranked = await self.reranker_service.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k,
            include_scores=True,
            use_mlx=self.use_mlx,
        )

        # Format results as (text, score) tuples expected by DSPy
        results = []
        for idx, score in reranked:
            results.append((documents[idx], float(score)))

        return results

    def retrieve(self, query_or_pred: Any, k: Optional[int] = None) -> List[Any]:
        """
        Retrieve passages for the given query.
        Required method for DSPy Retriever interface.

        Args:
            query_or_pred: Query string or DSPy prediction
            k: Number of results to return

        Returns:
            List of retrieved passages
        """
        results = self.forward(query_or_pred, k)
        return [passage for passage, _ in results]

    # Sync version for compatibility, calling the async one
    def __call__(self, query_or_pred: Any, k: Optional[int] = None) -> List[Any]:
        """
        Synchronous entry point for DSPy Retriever compatibility.
        Calls the async forward method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # This is a simplified way to handle nested loops if forward() is called from an async context.
                # Proper handling might require `nest_asyncio` or careful event loop management.
                # For now, assume it might be part of a larger async operation or a simple script.
                # This will block if the loop is already running and busy.
                # Consider guiding users to use an async-native approach with DSPy if possible.
                return asyncio.run(self.forward(query_or_pred, k))
            else:
                return asyncio.run(self.forward(query_or_pred, k))
        except RuntimeError:  # No event loop
            return asyncio.run(self.forward(query_or_pred, k))
        except Exception as e:  # Catching specific Exception and logging
            logger.error(f"DSPyReranker __call__ unexpected error: {e}", exc_info=True)
            raise  # Re-raise the exception after logging
