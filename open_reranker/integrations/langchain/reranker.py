from typing import List, Optional
import asyncio

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, ConfigDict

from open_reranker.core.logging import setup_logging
from open_reranker.services.reranker_service import RerankerService

logger = setup_logging()


class LangChainReranker(BaseRetriever):
    """
    LangChain-compatible reranker that wraps OpenReranker service.

    This class allows using OpenReranker as a retriever in LangChain pipelines.
    It can be used on top of another retriever to rerank the results.
    """

    # Use ConfigDict for Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Define Pydantic fields properly
    base_retriever: Optional[BaseRetriever] = Field(
        default=None, description="Base retriever to wrap"
    )
    model: str = Field(default="default", description="Model name to use for reranking")
    top_k: int = Field(default=10, description="Number of top results to keep")
    use_mlx: bool = Field(
        default=True, description="Whether to use MLX acceleration if available"
    )

    def __init__(
        self,
        base_retriever: Optional[BaseRetriever] = None,
        model: str = "default",
        top_k: int = 10,
        use_mlx: bool = True,
        **kwargs,
    ):
        """
        Initialize the LangChain Reranker.

        Args:
            base_retriever: Optional base retriever to wrap
            model: Model name to use for reranking
            top_k: Number of top results to keep
            use_mlx: Whether to use MLX acceleration if available
        """
        super().__init__(
            base_retriever=base_retriever,
            model=model,
            top_k=top_k,
            use_mlx=use_mlx,
            **kwargs,
        )
        # Set up the reranker service as a private attribute
        object.__setattr__(self, "_reranker_service", RerankerService())

        logger.info(f"Initialized LangChainReranker with model={model}, top_k={top_k}")

    @property
    def reranker_service(self) -> RerankerService:
        """Get the reranker service instance."""
        return getattr(self, "_reranker_service", RerankerService())

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronously get documents relevant to the query.

        Args:
            query: Query string
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # If we have a base retriever, use it to get initial results
        if self.base_retriever:
            # Check if the base retriever has an async method
            if hasattr(self.base_retriever, "aget_relevant_documents"):
                base_documents = await self.base_retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
            else:
                base_documents = self.base_retriever.get_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
        else:
            # No base retriever, nothing to rerank
            logger.warning(
                "No base retriever and no documents provided for aget_relevant_documents"
            )
            return []

        # If we have no documents, return empty results
        if not base_documents:
            return []

        # Extract text from documents
        document_texts = [doc.page_content for doc in base_documents]
        document_metadatas = [doc.metadata for doc in base_documents]

        # Rerank the documents
        reranked = await self.reranker_service.rerank(
            query=query,
            documents=document_texts,
            model=self.model,
            top_k=self.top_k,
            include_scores=True,
            use_mlx=self.use_mlx,
        )

        # Create reranked document list
        reranked_documents = []
        for idx, score in reranked:
            # Create a new Document with the original content and metadata
            # Add the reranker score to the metadata
            metadata = dict(document_metadatas[idx])
            metadata["reranker_score"] = float(score)

            reranked_documents.append(
                Document(page_content=document_texts[idx], metadata=metadata)
            )

        return reranked_documents

    # Keep the synchronous version for compatibility, but make it call the async one.
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get documents relevant to the query.

        Args:
            query: Query string
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # If called from a sync context that already has a running loop (e.g. Jupyter)
        if loop.is_running():
            # Create a new task to run the async method and wait for its result
            # task = asyncio.create_task(self._aget_relevant_documents(query, run_manager=run_manager))
            # return asyncio.run(task) # This is problematic in running loops
            # A more robust solution for nested loops might involve nest_asyncio or other patterns.
            # For simplicity here, we'll fall back to a direct call if we can't create a new loop.
            # This might still fail in some contexts. Users should prefer async native calls.
            try:
                # This is a simplified way to handle it.
                # If we're in a running loop, try to schedule and run to completion.
                # This is not foolproof for all nested asyncio scenarios.
                # logger.warning("Running async _aget_relevant_documents from a sync context with a running event loop. This might be unstable.")
                future = asyncio.ensure_future(self._aget_relevant_documents(query, run_manager=run_manager))
                # This is a simple way to block and get result if we are in a running loop already.
                # It might not be ideal for all scenarios.
                while not future.done():
                    # This is a placeholder for a more sophisticated wait if needed in some contexts.
                    # For many cases, the loop might process it quickly, or this might busy-wait briefly.
                    # Consider if a small sleep is needed or if the environment handles this.
                    pass # Potentially asyncio.sleep(0.001) if needed, but can add latency.
                return future.result()
            except Exception as e:
                logger.error(f"Error running _aget_relevant_documents in existing loop: {e}", exc_info=True)
                # Fallback or re-raise, depending on desired behavior
                raise
        else:
            return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))
