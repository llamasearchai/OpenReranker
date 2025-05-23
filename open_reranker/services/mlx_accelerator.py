import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import PyTorch for fallback
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import MLX conditionally - it's only available on macOS platforms
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from open_reranker.core.logging import setup_logging

logger = setup_logging()


class MLXAccelerator:
    """Service for MLX acceleration on Apple Silicon."""

    def __init__(self, device: str = "gpu"):
        """
        Initialize the MLX accelerator.

        Args:
            device: The device to use ("gpu" or "cpu")
        """
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Install it with 'pip install mlx'."
            )

        self.device = device
        self.model_cache = {}

        # Set the default compute device
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)

        logger.info(f"Initialized MLX accelerator on {device}")

    def convert_to_mlx_tensor(self, data: Any) -> Any:
        """
        Convert data to an MLX tensor.

        Args:
            data: The data to convert (numpy array, list, or pytorch tensor)

        Returns:
            An MLX tensor
        """
        if isinstance(data, mx.array):
            return data

        # Convert PyTorch tensor
        if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
            # It's likely a PyTorch tensor
            data = data.detach().cpu().numpy()

        # Convert NumPy array or list
        return mx.array(data)

    def convert_from_mlx_tensor(self, tensor: Any) -> np.ndarray:
        """
        Convert an MLX tensor to a NumPy array.

        Args:
            tensor: The MLX tensor to convert

        Returns:
            A NumPy array
        """
        if isinstance(tensor, mx.array):
            return np.array(tensor.tolist())
        return tensor

    def compute_cross_encoder_scores(
        self, cross_encoder, query: str, documents: List[str], batch_size: int = 16
    ) -> np.ndarray:
        """
        Compute cross-encoder scores using MLX acceleration.

        Args:
            cross_encoder: The cross-encoder model
            query: The search query
            documents: List of documents to score
            batch_size: Batch size for processing

        Returns:
            Array of scores for each document
        """
        # For now, fall back to PyTorch implementation
        # In a full MLX implementation, we would convert the model to MLX format
        logger.info(
            "Using PyTorch fallback for cross-encoder (MLX model conversion not implemented)"
        )

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for fallback")

        return cross_encoder.compute_scores(query, documents, batch_size)
