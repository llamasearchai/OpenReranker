"""DSPy integration for OpenReranker."""

try:
    from .reranker import DSPyReranker

    __all__ = ["DSPyReranker"]
except ImportError:
    __all__ = []
