"""Integrations module for OpenReranker."""

# Import integrations conditionally based on available dependencies
try:
    from .dspy.reranker import DSPyReranker

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

try:
    from .langchain.reranker import LangChainReranker

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

__all__ = []

if DSPY_AVAILABLE:
    __all__.append("DSPyReranker")

if LANGCHAIN_AVAILABLE:
    __all__.append("LangChainReranker")
