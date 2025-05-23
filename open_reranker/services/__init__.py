"""Services module for OpenReranker."""

from .mlx_accelerator import MLXAccelerator
from .reranker_service import RerankerService

# MLX accelerator is optional and platform-specific
try:
    from .mlx_accelerator import MLXAccelerator

    __all__ = ["RerankerService", "MLXAccelerator"]
except ImportError:
    __all__ = ["RerankerService", "MLXAccelerator"]
