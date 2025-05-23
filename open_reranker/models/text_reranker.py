from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from open_reranker.core.logging import setup_logging
from open_reranker.models.base import BaseReranker
from open_reranker.models.cross_encoder import CrossEncoder

logger = setup_logging()


class TextReranker(BaseReranker):
    """Text reranker model for reranking documents based on query relevance."""

    def __init__(self, model_name: str):
        """
        Initialize the text reranker model.

        Args:
            model_name: The model name or path
        """
        super().__init__(model_name)
        self.cross_encoder = CrossEncoder(model_name)
        self.batch_size = 32
        self.accelerator = None

    def set_accelerator(self, accelerator):
        """Set the MLX accelerator for optimized operations."""
        self.accelerator = accelerator
        self.cross_encoder.set_accelerator(accelerator)

    def compute_scores(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: The search query
            documents: List of documents to score

        Returns:
            Array of scores for each document
        """
        if self.accelerator:
            # Use MLX acceleration if available
            return self.accelerator.compute_cross_encoder_scores(
                self.cross_encoder, query, documents, self.batch_size
            )
        else:
            # Use regular PyTorch computation
            return self.cross_encoder.compute_scores(query, documents, self.batch_size)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of tuples (document_idx, score) sorted by relevance
        """
        scores = self.compute_scores(query, documents)

        # Get top-k results
        if top_k is None or top_k >= len(documents):
            top_k = len(documents)

        # Sort by score in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Create result tuples (idx, score)
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices]

        return results
