import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from open_reranker.core.cache import cache_manager
from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging
from open_reranker.core.monitoring import get_metrics, track_time
from open_reranker.models.code_reranker import CodeReranker
from open_reranker.models.table_reranker import TableReranker
from open_reranker.models.text_reranker import TextReranker
from open_reranker.services.mlx_accelerator import MLXAccelerator
from open_reranker.utils.text_utils import truncate_text

logger = setup_logging()


class RerankerService:
    """Service for reranking documents, code, and tables."""

    def __init__(self):
        """Initialize the reranker service."""
        self.text_rerankers: Dict[str, TextReranker] = {}
        self.code_reranker: Optional[CodeReranker] = None
        self.table_reranker: Optional[TableReranker] = None
        self.accelerator: Optional[MLXAccelerator] = None
        self.last_rerank_time: float = 0.0

    def set_accelerator(self, accelerator: MLXAccelerator):
        """Set the MLX accelerator for optimized operations."""
        self.accelerator = accelerator

    def _get_text_reranker(self, model: Union[str, Any]) -> TextReranker:
        """
        Get a cached text reranker or load it if not available.

        Args:
            model: The reranker model name or enum

        Returns:
            The loaded text reranker
        """
        model_str = str(model)

        # Use default model if not specified
        if model_str == "default":
            model_str = settings.DEFAULT_RERANKER_MODEL

        # Return cached model if available
        if model_str in self.text_rerankers:
            return self.text_rerankers[model_str]

        # Load model
        logger.info(f"Loading text reranker model: {model_str}")
        start_time = time.time()

        reranker = TextReranker(model_str)
        if self.accelerator:
            reranker.set_accelerator(self.accelerator)

        self.text_rerankers[model_str] = reranker

        # Record metrics
        load_time = time.time() - start_time
        get_metrics().model_load_duration.labels(model=model_str).observe(load_time)

        logger.info(f"Loaded text reranker model in {load_time:.2f} seconds")

        return reranker

    def _get_code_reranker(self) -> CodeReranker:
        """
        Get a cached code reranker or load it if not available.

        Returns:
            The loaded code reranker
        """
        if self.code_reranker is not None:
            return self.code_reranker

        # Load model
        logger.info(f"Loading code reranker model: {settings.CODE_RERANKER_MODEL}")
        start_time = time.time()

        self.code_reranker = CodeReranker(settings.CODE_RERANKER_MODEL)
        if self.accelerator:
            self.code_reranker.set_accelerator(self.accelerator)

        # Record metrics
        load_time = time.time() - start_time
        get_metrics().model_load_duration.labels(
            model=settings.CODE_RERANKER_MODEL
        ).observe(load_time)

        logger.info(f"Loaded code reranker model in {load_time:.2f} seconds")

        return self.code_reranker

    def _get_table_reranker(self) -> TableReranker:
        """
        Get a cached table reranker or load it if not available.

        Returns:
            The loaded table reranker
        """
        if self.table_reranker is not None:
            return self.table_reranker

        # Load model
        logger.info(f"Loading table reranker model: {settings.TABLE_RERANKER_MODEL}")
        start_time = time.time()

        self.table_reranker = TableReranker(settings.TABLE_RERANKER_MODEL)
        if self.accelerator:
            self.table_reranker.set_accelerator(self.accelerator)

        # Record metrics
        load_time = time.time() - start_time
        get_metrics().model_load_duration.labels(
            model=settings.TABLE_RERANKER_MODEL
        ).observe(load_time)

        logger.info(f"Loaded table reranker model in {load_time:.2f} seconds")

        return self.table_reranker

    @track_time("rerank", "rerank_duration")
    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Union[str, Any] = "default",
        top_k: Optional[int] = None,
        include_scores: bool = True,
        truncate_query: bool = True,
        truncate_document: bool = True,
        use_mlx: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            model: Reranker model to use
            top_k: Number of top documents to return
            include_scores: Whether to include scores in the response
            truncate_query: Whether to truncate the query if it's too long
            truncate_document: Whether to truncate documents if they're too long
            use_mlx: Whether to use MLX acceleration

        Returns:
            List of tuples (document_idx, score) sorted by relevance
        """
        start_time = time.time()

        # Handle empty documents case
        if not documents:
            return []

        # Truncate query if needed
        if truncate_query and len(query) > settings.MAX_QUERY_LENGTH:
            query = truncate_text(query, settings.MAX_QUERY_LENGTH)

        # Truncate documents if needed
        if truncate_document:
            documents = [
                (
                    truncate_text(doc, settings.MAX_DOCUMENT_LENGTH)
                    if len(doc) > settings.MAX_DOCUMENT_LENGTH
                    else doc
                )
                for doc in documents
            ]

        # Set up MLX accelerator if needed
        if use_mlx and settings.USE_MLX and self.accelerator is None:
            try:
                from open_reranker.services.mlx_accelerator import MLXAccelerator

                self.accelerator = MLXAccelerator()
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to initialize MLX accelerator: {str(e)}")

        # Get reranker
        reranker_instance = self._get_text_reranker(model)
        model_str = str(model)
        if model_str == "default":
            model_str = settings.DEFAULT_RERANKER_MODEL

        # Rerank documents
        # Check cache for scores first
        cached_scores = await cache_manager.get_model_scores(
            model=model_str, query=query, documents=documents
        )

        if cached_scores is not None:
            scores = np.array(cached_scores)
            logger.debug(
                f"Cache hit for model scores: model={model_str}, query='{query[:30]}...'"
            )
            get_metrics().cache_hits.labels(cache_type="model_scores").inc()
        else:
            logger.debug(
                f"Cache miss for model scores: model={model_str}, query='{query[:30]}...'"
            )
            get_metrics().cache_misses.labels(cache_type="model_scores").inc()
            scores = reranker_instance.compute_scores(query, documents)
            # Cache the computed scores
            await cache_manager.set_model_scores(
                model=model_str,
                query=query,
                documents=documents,
                scores=scores.tolist(),  # Convert numpy array to list for JSON serialization
            )

        # Get top-k results
        if top_k is None or top_k >= len(documents):
            top_k = len(documents)

        # Sort by score in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Create result tuples (idx, score)
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices]

        # Record timing
        self.last_rerank_time = time.time() - start_time

        return results
