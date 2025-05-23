from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseReranker(ABC):
    """Base class for reranker models."""

    def __init__(self, model_name: str):
        """
        Initialize the base reranker model.

        Args:
            model_name: The model name or path
        """
        self.model_name = model_name
        self.accelerator = None

    def set_accelerator(self, accelerator):
        """
        Set the MLX accelerator for optimized operations.

        Args:
            accelerator: The MLX accelerator instance
        """
        self.accelerator = accelerator

    @abstractmethod
    def compute_scores(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: The search query
            documents: List of documents to score

        Returns:
            Array of scores for each document
        """
        pass

    @abstractmethod
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
        pass
