from typing import List, Any, Optional, Tuple
import numpy as np
from open_reranker.models.base import BaseReranker
from open_reranker.core.logging import setup_logging
from open_reranker.models.cross_encoder import CrossEncoder
from open_reranker.utils.table_utils import format_table_for_reranking

logger = setup_logging()


class TableReranker(BaseReranker):
    """Table reranker model for reranking tables based on query relevance."""

    def __init__(self, model_name: str):
        """
        Initialize the table reranker model.

        Args:
            model_name: The model name or path
        """
        super().__init__(model_name)
        self.cross_encoder = CrossEncoder(model_name)
        self.batch_size = 8  # Smaller batch size for tables which can be very large

    def set_accelerator(self, accelerator):
        """Set the MLX accelerator for optimized operations."""
        self.accelerator = accelerator
        self.cross_encoder.set_accelerator(accelerator)

    def compute_scores(self, query: str, formatted_tables: List[str]) -> np.ndarray:
        """
        Compute relevance scores for query-table pairs.

        Args:
            query: The search query
            formatted_tables: List of formatted table texts

        Returns:
            Array of scores for each table
        """
        # Use MLX acceleration if available
        if self.accelerator:
            return self.accelerator.compute_cross_encoder_scores(
                self.cross_encoder, query, formatted_tables, self.batch_size
            )
        else:
            return self.cross_encoder.compute_scores(
                query, formatted_tables, self.batch_size
            )

    def rerank(
        self,
        query: str,
        tables: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank tables based on their relevance to the query.

        Args:
            query: The search query
            tables: List of tables to rerank
            top_k: Number of top tables to return

        Returns:
            List of tuples (table_idx, score) sorted by relevance
        """
        # Format tables for reranking
        formatted_tables = []
        for table in tables:
            if hasattr(table, "headers") and hasattr(table, "rows"):
                # For custom TableDocument objects
                formatted_tables.append(
                    format_table_for_reranking(
                        table.headers,
                        table.rows,
                        table_name=getattr(table, "table_name", None),
                    )
                )
            elif isinstance(table, dict) and "headers" in table and "rows" in table:
                # For dict format
                formatted_tables.append(
                    format_table_for_reranking(
                        table["headers"],
                        table["rows"],
                        table_name=table.get("table_name"),
                    )
                )
            else:
                # Handle as string if not in expected format
                formatted_tables.append(str(table))

        # Compute scores
        scores = self.compute_scores(query, formatted_tables)

        # Get top-k results
        if top_k is None or top_k >= len(tables):
            top_k = len(tables)

        # Sort by score in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Create result tuples (idx, score)
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices]

        return results
