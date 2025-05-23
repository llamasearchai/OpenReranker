"""LangChain integration for OpenReranker."""

try:
    from .reranker import LangChainReranker

    __all__ = ["LangChainReranker"]
except ImportError:
    __all__ = []
