from typing import List, Optional, Tuple

import numpy as np

from open_reranker.core.logging import setup_logging
from open_reranker.models.base import BaseReranker
from open_reranker.models.cross_encoder import CrossEncoder
from open_reranker.utils.code_utils import format_code_for_reranking

logger = setup_logging()


class CodeReranker(BaseReranker):
    """Code reranker model for reranking code documents based on query relevance."""

    def __init__(self, model_name: str):
        """
        Initialize the code reranker model.

        Args:
            model_name: The model name or path
        """
        super().__init__(model_name)
        self.cross_encoder = CrossEncoder(model_name)
        self.batch_size = 16  # Smaller batch size for code which tends to be longer

    def set_accelerator(self, accelerator):
        """Set the MLX accelerator for optimized operations."""
        self.accelerator = accelerator
        self.cross_encoder.set_accelerator(accelerator)

    def compute_scores(
        self,
        query: str,
        code_documents: List[str],
        languages: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Compute relevance scores for query-code pairs.

        Args:
            query: The search query
            code_documents: List of code documents to score
            languages: Optional list of programming languages for each code document

        Returns:
            Array of scores for each code document
        """
        # Format code for reranking (language-specific formatting)
        formatted_code = [
            format_code_for_reranking(code, lang)
            for code, lang in zip(
                code_documents, languages or [None] * len(code_documents)
            )
        ]

        # Use MLX acceleration if available
        if self.accelerator:
            return self.accelerator.compute_cross_encoder_scores(
                self.cross_encoder, query, formatted_code, self.batch_size
            )
        else:
            return self.cross_encoder.compute_scores(
                query, formatted_code, self.batch_size
            )

    def rerank(
        self,
        query: str,
        code_documents: List[str],
        languages: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank code documents based on their relevance to the query.

        Args:
            query: The search query
            code_documents: List of code documents to rerank
            languages: Optional list of programming languages for each code document
            top_k: Number of top code documents to return

        Returns:
            List of tuples (document_idx, score) sorted by relevance
        """
        # Ensure languages list is appropriate length
        if languages is None:
            languages = [None] * len(code_documents)

        # Compute scores
        scores = self.compute_scores(query, code_documents, languages)

        # Get top-k results
        if top_k is None or top_k >= len(code_documents):
            top_k = len(code_documents)

        # Sort by score in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Create result tuples (idx, score)
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices]

        return results
