from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from open_reranker.core.logging import setup_logging

logger = setup_logging()


class CrossEncoder:
    """
    Cross-encoder model for reranking.

    A cross-encoder processes query and document together as a single sequence,
    allowing direct modeling of the query-document interaction.
    """

    def __init__(self, model_name: str):
        """
        Initialize the cross-encoder model.

        Args:
            model_name: The model name or path
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            logger.info(
                f"Loaded cross-encoder model {self.model_name} on {self.device}"
            )
        except Exception as e:
            logger.error(
                f"Error loading cross-encoder model {self.model_name}: {str(e)}"
            )
            raise

    def set_accelerator(self, accelerator):
        """Set the MLX accelerator for optimized operations."""
        self.accelerator = accelerator

    def compute_scores(
        self, query: str, documents: List[str], batch_size: int = 16
    ) -> np.ndarray:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: The search query
            documents: List of documents to score
            batch_size: Batch size for processing

        Returns:
            Array of scores for each document
        """
        # Prepare pairs of (query, document)
        pairs = [(query, doc) for doc in documents]

        # Initialize scores array
        scores = np.zeros(len(documents))

        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]

            # Tokenize inputs
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}

            # Compute scores
            with torch.no_grad():
                outputs = self.model(**features)
                batch_scores = outputs.logits

                # Handle different model output shapes
                if len(batch_scores.shape) > 1 and batch_scores.shape[1] > 1:
                    # Multi-class model, take the positive class score
                    batch_scores = batch_scores[:, 1]
                elif len(batch_scores.shape) > 1:
                    # Single class model with batch dimension
                    batch_scores = batch_scores.squeeze(-1)
                # For 1D tensors, use as-is

                # Convert to numpy and store
                batch_scores_np = batch_scores.cpu().numpy()
                scores[i : i + len(batch_pairs)] = batch_scores_np

        return scores 