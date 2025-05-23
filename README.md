# OpenReranker: High-Performance Neural Search Reranking

OpenReranker is an enterprise-grade, open-source reranking service designed to significantly enhance search result relevance. It leverages state-of-the-art cross-encoder models and offers seamless integration with popular AI frameworks like LangChain and DSPy. Optimized for performance, including MLX acceleration for Apple Silicon, OpenReranker is built for demanding production environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/open-reranker.svg)](https://pypi.org/project/open-reranker/)
[![PyPI Version](https://img.shields.io/pypi/v/open-reranker.svg)](https://pypi.org/project/open-reranker/)
[![Build Status](https://github.com/llamasearchai/OpenReranker/actions/workflows/ci.yml/badge.svg)](https://github.com/llamasearchai/OpenReranker/actions/workflows/ci.yml)
[![Code Coverage](https://codecov.io/gh/llamasearchai/OpenReranker/branch/main/graph/badge.svg)](https://codecov.io/gh/llamasearchai/OpenReranker)

## Table of Contents

- [Key Features](#key-features)
- [Core Concepts](#core-concepts)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Environment Configuration](#environment-configuration)
  - [Running the Server](#running-the-server)
- [Usage Examples](#usage-examples)
  - [API Usage: Reranking Documents](#api-usage-reranking-documents)
  - [LangChain Integration](#langchain-integration)
  - [DSPy Integration](#dspy-integration)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Redis Setup](#redis-setup)
- [Models](#models)
  - [Supported Models](#supported-models)
  - [Custom Models](#custom-models)
- [Advanced Features](#advanced-features)
  - [Authentication and Authorization](#authentication-and-authorization)
  - [Rate Limiting](#rate-limiting)
  - [Caching Strategies](#caching-strategies)
  - [MLX Acceleration](#mlx-acceleration)
  - [Monitoring and Logging](#monitoring-and-logging)
- [Architecture Overview](#architecture-overview)
- [Development](#development)
  - [Setup](#setup)
  - [Testing](#testing)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Contributing](#contributing)
- [License](#license)

## Key Features

-   **State-of-the-Art Reranking:** Employs advanced cross-encoder models to deliver superior search relevance across various data types.
-   **Framework Agnostic & Integrated:** Offers direct, efficient class-based integrations with LangChain and DSPy, alongside a robust REST API.
-   **Specialized Model Support:** Architected to support distinct rerankers optimized for text, source code, and tabular data (future enhancements for code/table models).
-   **Performance-Driven:** Features MLX acceleration for significant speedups on Apple Silicon hardware.
-   **Asynchronous by Design:** Built with FastAPI for high-throughput, non-blocking API operations.
-   **Production-Grade Reliability:**
    -   **Comprehensive Logging:** Structured JSON logging for easy analysis and integration with log management systems.
    -   **Prometheus Metrics:** Exposes detailed metrics for performance monitoring and alerting.
    -   **Secure Authentication:** JWT-based authentication for protecting API endpoints.
    -   **Tiered Rate Limiting:** Sophisticated rate limiting (QPS, RPM, TPM) with Redis or in-memory backing, configurable per user tier.
    -   **Intelligent Caching:** Multi-level caching (full responses and model scores) using Redis or in-memory stores to reduce latency and computational load.
    -   **Scalability:** Configurable worker counts for horizontal scaling.

## Core Concepts

**Reranking:** The process of taking an initial set of search results (often from a traditional keyword search or a dense retriever like an embedding model) and re-ordering them based on a more computationally intensive but accurate relevance model. Cross-encoders, used by OpenReranker, achieve high accuracy by jointly encoding the query and each document.

**Cross-Encoders:** Neural models that take a query and a document as a single input to produce a relevance score. This allows for deep interaction modeling between the query and document, leading to higher accuracy than bi-encoder (embedding-based) approaches for final-stage ranking.

**MLX Acceleration:** A framework for machine learning on Apple Silicon, providing significant performance improvements for supported models by leveraging the unified memory architecture and Apple's Neural Engine.

## Installation

### Using pip

```bash
pip install open-reranker
```
This will install the core package. For optional dependencies related to integrations or MLX:
```bash
pip install open-reranker[integrations,mlx]
```

### From Source (for development or contributions)

```bash
git clone https://github.com/llamasearchai/OpenReranker.git
cd OpenReranker

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Install dependencies
# Ensure pip is up-to-date for pyproject.toml extras
python -m pip install --upgrade pip
# Install core, development, integrations, and MLX dependencies
pip install -e ".[dev,integrations,mlx]"
# For core dependencies only:
# pip install -e .
```

## Quick Start

### 1. Environment Configuration

Copy the `.env.example` file to `.env` and customize the settings. This file contains all configurable parameters for the service.

```bash
cp .env.example .env
# Open .env and modify settings such as SECRET_KEY, REDIS_URL (if used), etc.
```
Refer to the [Configuration](#configuration) section for details on all available environment variables.

### 2. Running the Server

You can run OpenReranker using Uvicorn. The number of workers can be configured via the `OPEN_RERANKER_NUM_WORKERS` environment variable or the `--workers` CLI flag.

**For development (with auto-reload):**
```bash
uvicorn open_reranker.main:app --reload --host 0.0.0.0 --port 8000
```

**For a single instance production/staging deployment:**
```bash
uvicorn open_reranker.main:app --host 0.0.0.0 --port 8000 --workers 4
```
(Adjust `--workers 4` based on your server's CPU cores.)

**For multi-process production, consider Gunicorn with Uvicorn workers:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker open_reranker.main:app -b 0.0.0.0:8000
```
(Ensure Gunicorn is installed: `pip install gunicorn`)

## Usage Examples

### API Usage: Reranking Documents

The primary endpoint for reranking is `POST /api/v1/rerank`.

```python
import requests
import json

API_URL = "http://localhost:8000/api/v1/rerank"

query = "What is the function of chlorophyll in plants?"
documents = [
    {"id": "doc1", "text": "Chlorophyll is a green pigment found in plants that absorbs light energy for photosynthesis."},
    {"id": "doc2", "text": "Plants use photosynthesis to convert carbon dioxide and water into glucose and oxygen."},
    {"id": "doc3", "text": "The Earth orbits around the Sun at an average distance of 149.6 million kilometers."}
]

payload = {
    "query": query,
    "documents": documents,
    "model": "default",  # Or specify a model name like "jinaai/jina-reranker-v2-base-en"
    "top_k": 2,
    "include_scores": True
}

# If authentication is enabled (OPEN_RERANKER_AUTH_ENABLED=True in .env),
# obtain a JWT token and include it in the headers:
# HEADERS = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
# response = requests.post(API_URL, json=payload, headers=HEADERS)

# Assuming auth is disabled for this example or token is handled:
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    results = response.json()
    print(f"Query: {results['query']}")
    print(f"Processing Time: {results['timing']['total_processing_time']:.4f} seconds (Reranking: {results['timing']['reranking_time']:.4f}s)")
    print(f"Cached Response: {results['metadata'].get('cached_response', False)}")
    print(f"Cached Model Scores: {results['metadata'].get('cached_model_scores', False)}")
    
    print("\nRanked Results:")
    for i, doc in enumerate(results["results"]):
        print(f"{i+1}. Score: {doc['score']:.4f} - ID: {doc['id']} - Text: {doc['text'][:100]}...")
elif response.status_code == 401:
    print(f"Authentication Error: {response.status_code} - {response.text}")
elif response.status_code == 429:
    print(f"Rate Limit Error: {response.status_code} - {response.text}")
else:
    print(f"Error: {response.status_code} - {response.text}")

```

### LangChain Integration

OpenReranker provides a `LangChainReranker` class that integrates directly with LangChain's retrieval components, using the `RerankerService` internally for efficiency (no HTTP overhead). It supports asynchronous operations.

```python
import asyncio
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

# Example of a simple custom retriever for demonstration
class MySimpleRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Synchronous version
        print(f"(MySimpleRetriever sync called for: '{query}')")
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Asynchronous version
        print(f"(MySimpleRetriever async called for: '{query}')")
        await asyncio.sleep(0.01) # Simulate async work
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]

from open_reranker.integrations.langchain import LangChainReranker

# Sample documents for the base retriever
base_documents = [
    Document(page_content="Document 1 about science and chlorophyll."),
    Document(page_content="Document 2 about history and empires."),
    Document(page_content="Document 3 details biology and plants, specifically chlorophyll function.")
]
base_retriever = MySimpleRetriever(docs=base_documents)

# Initialize the LangChainReranker
# This uses the RerankerService directly, not an HTTP endpoint.
open_reranker_retriever = LangChainReranker(
    base_retriever=base_retriever,
    model="default",  # Uses the default model configured in OpenReranker's RerankerService
    top_k=2
)

async def run_langchain_example():
    query = "Tell me about science and plants focusing on chlorophyll"
    
    print("\n--- LangChain Asynchronous Reranking ---")
    async_results = await open_reranker_retriever.aget_relevant_documents(query)
    print("Async Reranked Results:")
    for doc in async_results:
        print(f"Score: {doc.metadata.get('reranker_score', 0.0):.4f} - Content: {doc.page_content}")

    print("\n--- LangChain Synchronous Reranking ---")
    # The synchronous version internally uses asyncio.run to call the async logic
    sync_results = open_reranker_retriever.get_relevant_documents(query)
    print("Sync Reranked Results (via async wrapper):")
    for doc in sync_results:
        print(f"Score: {doc.metadata.get('reranker_score', 0.0):.4f} - Content: {doc.page_content}")

if __name__ == "__main__":
    # Ensure OpenReranker is installed and its dependencies (like transformers) are available.
    # This example can be run from the root of the OpenReranker project or if PYTHONPATH is set.
    # asyncio.run(run_langchain_example())
    print("LangChain example ready. To run: `asyncio.run(run_langchain_example())` in a suitable async environment.")
```

### DSPy Integration

The `DSPyReranker` module allows OpenReranker to be used as a reranking step within DSPy programs. It also utilizes the `RerankerService` directly.

```python
import asyncio
import dspy
from typing import List, Union, Any, Optional

# Example of a simple custom DSPy Retrieval Module (RM) for demonstration
class MySimpleDSPyRM(dspy.Retrieve):
    def __init__(self, docs: List[str], k: int = 3):
        super().__init__(k=k)
        # Store documents as dspy.Passage objects
        self.dspy_docs = [dspy.Passage(long_text=doc) for doc in docs]

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        k_to_use = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        all_passages = []
        for query_text in queries:
            # Simple keyword match for demonstration
            matched_passages = [doc for doc in self.dspy_docs if query_text.lower() in doc.long_text.lower()]
            all_passages.extend(matched_passages)
        
        # DSPy Retrieve.forward should return a dspy.Prediction containing a list of passages
        # For simplicity, we're not sorting by any initial score here.
        return dspy.Prediction(passages=all_passages[:k_to_use])


from open_reranker.integrations.dspy import DSPyReranker

sample_texts_for_dspy = [
    "Chlorophyll is vital for photosynthesis, the process plants use to make food.",
    "The ancient Roman Empire had a significant impact on Western civilization.",
    "Quantum entanglement is a phenomenon in quantum physics.",
    "Advanced research on chlorophyll explores its role in light harvesting."
]

# Create a base DSPy retrieval module
base_dspy_retriever = MySimpleDSPyRM(docs=sample_texts_for_dspy, k=10)

# Initialize DSPyReranker
# This wraps the base_dspy_retriever and reranks its results.
dspy_reranker_module = DSPyReranker(
    base_retriever=base_dspy_retriever,
    model="default",  # Corresponds to OPEN_RERANKER_DEFAULT_RERANKER_MODEL
    top_k=2
)

# Example DSPy Program using the reranker
class SimpleRAGWithDSPyReranker(dspy.Module):
    def __init__(self):
        super().__init__()
        # DSPyReranker itself acts as a dspy.Retrieve module
        self.retrieve_and_rerank = dspy_reranker_module
        # In a full RAG, you'd have a generation model here:
        # self.generate_answer = dspy.Predict("context, question -> answer")

    def forward(self, query: str) -> dspy.Prediction:
        # The DSPyReranker module, when called, will first use its base_retriever
        # to get initial documents, then rerank them, and return a dspy.Prediction
        # containing the reranked dspy.Passage objects.
        reranked_context = self.retrieve_and_rerank(query)
        
        # For demonstration, we return the reranked context.
        # A real RAG would use this context for answer generation:
        # prediction = self.generate_answer(context=reranked_context.passages, question=query)
        return reranked_context

async def run_dspy_example():
    query = "What is chlorophyll and its role in plants?"

    print("\n--- DSPy Integration Test ---")
    
    # DSPy modules are typically used synchronously.
    # The DSPyReranker's __call__ method handles async operations internally if needed.
    
    # Configure a dummy LM for DSPy if not already set globally (for Predict modules)
    # For this example, since we only use Retrieve, it's not strictly necessary.
    # if not dspy.settings.lm:
    #     dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=100))


    rag_program_instance = SimpleRAGWithDSPyReranker()
    
    # Execute the RAG program
    final_prediction = rag_program_instance(query=query)
    
    print(f"Reranked Passages for query: '{query}':")
    if final_prediction.passages:
        for i, passage in enumerate(final_prediction.passages):
            # Scores are added to passage.score by DSPyReranker
            score = passage.score if hasattr(passage, 'score') and passage.score is not None else "N/A"
            print(f"{i+1}. Score: {score:.4f} - Text: {passage.long_text}")
    else:
        print("No passages returned after reranking.")

if __name__ == "__main__":
    # Ensure OpenReranker & DSPy are installed and dependencies are available.
    # Needs to be run from project root or with PYTHONPATH set. Models must be loadable.
    # asyncio.run(run_dspy_example())
    print("DSPy example ready. To run: `asyncio.run(run_dspy_example())` in a suitable async environment.")

```

## API Endpoints

Detailed list of API endpoints:

-   **`POST /api/v1/rerank`**: Reranks a list of documents for a given query.
    -   **Request Body**: `RerankerRequest` model (query, documents, model, top_k, etc.)
    -   **Response**: `RerankerResponse` model (ranked documents, scores, timing, metadata)
    -   Supports response caching and model score caching.
    -   Asynchronous operation.
-   **`POST /api/v1/rerank/batch`**: Reranks multiple sets of documents for multiple queries in a single request.
    -   **Request Body**: `BatchRerankerRequest` model.
    -   **Response**: `BatchRerankerResponse` model.
    -   Requires a specific user tier (default: "pro", configurable).
    -   Asynchronous operation.
-   **`POST /api/v1/integrations/dspy`**: (Legacy HTTP Endpoint) Provides an HTTP interface for DSPy integration.
    -   The `DSPyReranker` class (direct service usage) is the preferred method.
    -   Asynchronous operation.
-   **`POST /api/v1/integrations/langchain`**: (Legacy HTTP Endpoint) Provides an HTTP interface for LangChain integration.
    -   The `LangChainReranker` class (direct service usage) is the preferred method.
    -   Asynchronous operation.
-   **`GET /`**: Health check endpoint. Returns `{"status": "healthy"}`.
-   **`GET /metrics`**: Prometheus metrics endpoint. Exposes various operational metrics.
-   **`DELETE /api/v1/cache`**: Clears all cached data (full responses and model scores).
    -   Requires a specific user tier (default: "enterprise", configurable).

*Authentication (JWT Bearer token) is enforced if `OPEN_RERANKER_AUTH_ENABLED` is true.*

## Configuration

OpenReranker is configured primarily through environment variables. Create a `.env` file in the project root by copying `.env.example`.

### Environment Variables

Below are key environment variables. Refer to `.env.example` for a comprehensive list and default values.

**General API Settings:**
-   `OPEN_RERANKER_API_PREFIX`: Base path for API routes (Default: `/api/v1`).
-   `OPEN_RERANKER_DEBUG`: Enable FastAPI debug mode (Default: `False`).
-   `OPEN_RERANKER_HOST`: Host address to bind (Default: `0.0.0.0`).
-   `OPEN_RERANKER_PORT`: Server port (Default: `8000`).

**CORS (Cross-Origin Resource Sharing):**
-   `OPEN_RERANKER_CORS_ORIGINS`: Comma-separated list of allowed origins. `*` for all. (Default: `*`).

**Authentication:**
-   `OPEN_RERANKER_AUTH_ENABLED`: Enable JWT authentication (Default: `True`).
-   `OPEN_RERANKER_SECRET_KEY`: **Critical for production.** Secret key for signing JWTs.
-   `OPEN_RERANKER_ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiry (Default: `10080` - 7 days).
-   `OPEN_RERANKER_USER_TIERS`: Comma-separated list of defined user tiers (Default: `free,pro,enterprise`).
-   `OPEN_RERANKER_BATCH_ENDPOINT_TIER`: Required tier for `/rerank/batch` (Default: `pro`).
-   `OPEN_RERANKER_CACHE_ENDPOINT_TIER`: Required tier for `DELETE /cache` (Default: `enterprise`).

**Redis for Caching & Rate Limiting:**
-   `OPEN_RERANKER_REDIS_URL`: Redis connection URL (e.g., `redis://localhost:6379/0`). If unset, in-memory stores are used.
-   `OPEN_RERANKER_CACHE_ENABLED`: Enable response caching (Default: `True`).
-   `OPEN_RERANKER_CACHE_TTL`: TTL for full API responses in seconds (Default: `3600` - 1 hour).
-   `OPEN_RERANKER_MODEL_SCORE_CACHE_TTL`: TTL for individual model scores (Default: `86400` - 1 day).

**Rate Limiting:**
-   `OPEN_RERANKER_RATE_LIMIT_QPS`: Queries Per Second for the base tier (Default: `100`).
-   `OPEN_RERANKER_RATE_LIMIT_RPM`: Requests Per Minute for the base tier (Default: `6000`).
-   `OPEN_RERANKER_RATE_LIMIT_TPM`: Tokens Per Minute for the base tier (Default: `1000000`).
-   `OPEN_RERANKER_COUNT_TOKENS`: Enable token counting for TPM rate limiting (Default: `True`).
-   `OPEN_RERANKER_PRO_TIER_SCALE_FACTOR`: Multiplier for 'pro' tier limits (Default: `2.0`).
-   `OPEN_RERANKER_ENTERPRISE_TIER_SCALE_FACTOR`: Multiplier for 'enterprise' tier limits (Default: `5.0`).

**Model Configuration:**
-   `OPEN_RERANKER_DEFAULT_RERANKER_MODEL`: Default model for text reranking (Default: `jinaai/jina-reranker-v2-base-en`).
-   `OPEN_RERANKER_CODE_RERANKER_MODEL`: Model for code reranking (Default: `jinaai/jina-reranker-v2-base-code`).
-   `OPEN_RERANKER_TABLE_RERANKER_MODEL`: Model for table reranking.
-   `OPEN_RERANKER_TIER_MODELS`: JSON string to map tiers to specific models (e.g., `{"free": "model_A", "pro": "model_B"}`). (Default: `{}`).

**MLX Acceleration (Apple Silicon):**
-   `OPEN_RERANKER_USE_MLX`: Attempt to use MLX if on Darwin (Default: `True`).
-   `OPEN_RERANKER_MLX_DEVICE`: `gpu` or `cpu` for MLX (Default: `gpu`).

**Batch Processing & Performance:**
-   `OPEN_RERANKER_MAX_BATCH_SIZE`: Max documents processed by the model in one internal batch (Default: `32`).
-   `OPEN_RERANKER_MAX_DOCUMENTS_PER_QUERY`: Max documents accepted per query in API calls (Default: `1000`).
-   `OPEN_RERANKER_MAX_QUERY_LENGTH`: Max characters for query truncation (Default: `1024`).
-   `OPEN_RERANKER_MAX_DOCUMENT_LENGTH`: Max characters for document truncation (Default: `8192`).
-   `OPEN_RERANKER_NUM_WORKERS`: Number of Uvicorn workers (Default: `4`).

**Monitoring:**
-   `OPEN_RERANKER_ENABLE_MONITORING`: Enable Prometheus metrics (Default: `True`).

### Redis Setup

For production or multi-worker deployments, using Redis is highly recommended for distributed caching and rate limiting.
1.  **Install Redis:** Follow official instructions at [redis.io/docs/getting-started/installation](https://redis.io/docs/getting-started/installation/).
2.  **Start Redis Server:** Typically `redis-server`.
3.  **Configure URL:** Set `OPEN_RERANKER_REDIS_URL` in your `.env` file (e.g., `OPEN_RERANKER_REDIS_URL="redis://localhost:6379/0"`).

If `OPEN_RERANKER_REDIS_URL` is not provided, the service will fall back to in-memory stores, which are not suitable for distributed environments.

## Models

### Supported Models

OpenReranker is designed to work with a variety of cross-encoder models from the Hugging Face Hub. Default models are configured for general-purpose text, code, and (planned) table reranking.

-   **Default Text Reranker:** `jinaai/jina-reranker-v2-base-en`
-   **Default Code Reranker:** `jinaai/jina-reranker-v2-base-code`
-   **Default Table Reranker:** (To be determined - currently uses a placeholder)

These defaults can be overridden via environment variables.

### Custom Models

You can use any compatible cross-encoder model from Hugging Face by:
1.  Setting the environment variables (e.g., `OPEN_RERANKER_DEFAULT_RERANKER_MODEL="your-org/your-model-name"`).
2.  Passing the `model` parameter in the API request payload for the `/rerank` endpoint.
3.  Specifying tier-specific models using `OPEN_RERANKER_TIER_MODELS`.

The model must be a `AutoModelForSequenceClassification` compatible model.

## Advanced Features

### Authentication and Authorization

When `OPEN_RERANKER_AUTH_ENABLED=True`:
-   API endpoints are protected and require a JWT Bearer token in the `Authorization` header.
-   JWTs are signed and verified using the `OPEN_RERANKER_SECRET_KEY`.
-   **JWT Claims:**
    -   `sub` (Subject): A unique identifier for the user or client. Used as the client ID for rate limiting.
    -   `tier`: User tier (e.g., "free", "pro", "enterprise"). Tiers are defined in `OPEN_RERANKER_USER_TIERS`.
    -   `exp`: Standard JWT expiration timestamp.
-   **Tier-Based Access Control:**
    -   The `/api/v1/rerank/batch` endpoint requires a tier specified by `OPEN_RERANKER_BATCH_ENDPOINT_TIER` (default: "pro") or higher.
    -   The `DELETE /api/v1/cache` endpoint requires a tier specified by `OPEN_RERANKER_CACHE_ENDPOINT_TIER` (default: "enterprise") or higher.
    -   Tier hierarchy is: `free` < `pro` < `enterprise`.

### Rate Limiting

OpenReranker implements a sophisticated, tiered rate-limiting system:
-   **Client Identification:** Uses the `sub` claim from the JWT if auth is enabled and valid. Falls back to the client's IP address otherwise.
-   **Limit Types:**
    -   Queries Per Second (QPS)
    -   Requests Per Minute (RPM)
    -   Tokens Per Minute (TPM) - calculated if `OPEN_RERANKER_COUNT_TOKENS=True`. Token count includes the sum of query length and all document lengths in a request.
-   **Tiered Limits:** Base limits are defined for the "free" tier (e.g., `OPEN_RERANKER_RATE_LIMIT_QPS`). "Pro" and "enterprise" tiers have their limits scaled by `OPEN_RERANKER_PRO_TIER_SCALE_FACTOR` and `OPEN_RERANKER_ENTERPRISE_TIER_SCALE_FACTOR` respectively.
-   **Storage Backend:** Uses Redis if `OPEN_RERANKER_REDIS_URL` is configured, enabling distributed rate limiting. Falls back to in-memory storage (per-process) otherwise.

### Caching Strategies

Two levels of caching are implemented to optimize performance:
1.  **Full Response Caching:**
    -   Caches the entire JSON response for `/api/v1/rerank` requests.
    -   Cache key includes: model name, query, all document texts, and `top_k`.
    -   TTL is controlled by `OPEN_RERANKER_CACHE_TTL`.
    -   Enabled if `OPEN_RERANKER_CACHE_ENABLED=True`.
2.  **Model Score Caching:**
    -   Caches the raw relevance scores (logits) computed by the underlying cross-encoder model for each query-document pair.
    -   This is a lower-level cache used by `RerankerService`.
    -   Cache key includes: model name, query, and individual document text.
    -   TTL is controlled by `OPEN_RERANKER_MODEL_SCORE_CACHE_TTL`.
    -   Always enabled if a cache backend (Redis or in-memory) is available.
-   **Storage Backend:** Uses Redis for both cache types if `OPEN_RERANKER_REDIS_URL` is set and caching is enabled. Falls back to in-memory LRU caches otherwise.
-   **Cache Invalidation:** The `DELETE /api/v1/cache` endpoint allows clearing all cached data (both full responses and model scores). Requires appropriate tier access.

### MLX Acceleration

-   If `OPEN_RERANKER_USE_MLX=True` (default) and the service is running on macOS with Apple Silicon, OpenReranker will attempt to use MLX for model inference.
-   This can provide substantial performance gains for supported Hugging Face Transformer models.
-   The device for MLX (GPU or CPU) can be specified with `OPEN_RERANKER_MLX_DEVICE` (default: `gpu`).
-   If MLX initialization fails or is disabled, the service gracefully falls back to standard PyTorch-based inference.

### Monitoring and Logging

-   **Prometheus Metrics:** If `OPEN_RERANKER_ENABLE_MONITORING=True` (default), a `/metrics` endpoint is exposed, providing detailed operational metrics compatible with Prometheus. Metrics include request counts, latencies, error rates, cache performance, model load times, and rate limit statuses.
-   **Structured Logging:** All log output is in JSON format, making it easy to parse, search, and integrate with centralized logging systems (e.g., ELK stack, Splunk). Logs include timestamps, log levels, request details (path, method, client IP), processing durations, and status codes.

## Architecture Overview

OpenReranker is built using FastAPI and follows a modular, service-oriented architecture:

1.  **API Layer (`open_reranker.api`):**
    -   Defines FastAPI routers, request/response models (Pydantic), and endpoint logic.
    -   Handles request validation, authentication, rate limiting, and response caching.
    -   Delegates core reranking tasks to the `RerankerService`.
2.  **Core Services (`open_reranker.services`):**
    -   `RerankerService`: Orchestrates the reranking process, manages model loading, and handles model score caching.
    -   `MLXAccelerator`: Provides an abstraction for MLX-based inference if available.
3.  **Models (`open_reranker.models`):**
    -   `BaseReranker`: Abstract base class for reranker models.
    -   `TextReranker`, `CodeReranker`, `TableReranker`: Concrete implementations for different data types.
    -   `CrossEncoder`: Wrapper around Hugging Face Transformer models for computing relevance scores.
4.  **Integrations (`open_reranker.integrations`):**
    -   `LangChainReranker`: A LangChain `BaseRetriever` compatible class.
    -   `DSPyReranker`: A DSPy `Retrieve` compatible module.
    -   These integrations use the `RerankerService` directly for efficiency.
5.  **Core Components (`open_reranker.core`):**
    -   `config.py`: Handles application settings via Pydantic and environment variables.
    -   `auth.py`: JWT authentication and tier-based authorization logic.
    -   `cache.py`: Caching mechanisms (Redis and in-memory).
    -   `rate_limiting.py`: Tiered rate limiting logic (Redis and in-memory).
    -   `logging.py`: Structured JSON logging setup.
    -   `monitoring.py`: Prometheus metrics setup.
6.  **Utilities (`open_reranker.utils`):**
    -   Helper functions for text processing, code formatting, table serialization, etc.

**Data Flow (Rerank Request):**
Request -> FastAPI Endpoint -> Auth Middleware -> Rate Limiting Middleware -> Cache Check (Full Response) -> `RerankerService.rerank()` -> Model Score Cache Check -> Model Inference (`CrossEncoder.compute_scores()` potentially via `MLXAccelerator`) -> Score Caching -> Sorting & Top-K -> Response Formatting -> Full Response Caching -> Response

## Development

### Setup

Follow the "From Source" installation instructions. Key steps:
1.  Clone repository.
2.  Create and activate a Python virtual environment.
3.  Install dependencies: `pip install -e ".[dev,integrations,mlx]"`
4.  Copy `.env.example` to `.env` and configure as needed.

### Testing

OpenReranker uses `pytest` for testing.
-   Run all tests:
    ```bash
    python -m pytest
    ```
-   Run tests with coverage:
    ```bash
    python -m pytest --cov=open_reranker --cov-report=term-missing --cov-report=xml
    ```
-   Specific tests:
    ```bash
    python -m pytest tests/test_api.py::TestAPI::test_rerank_endpoint_success
    ```

Tests cover API endpoints, service logic, model interactions (mocked), and integrations.

### Pre-commit Hooks

Pre-commit hooks are configured using `pre-commit` to ensure code quality and consistency (black, isort, flake8, mypy).
1.  Install pre-commit: `pip install pre-commit`
2.  Install hooks: `pre-commit install`
Hooks will run automatically before each commit. You can also run them manually: `pre-commit run --all-files`

## Contributing

Contributions are welcome! Please follow these general guidelines:
1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/issue-description`.
3.  **Make your changes.** Ensure you add or update tests for your changes.
4.  **Follow coding standards:** Run `pre-commit run --all-files` to format and lint your code.
5.  **Write clear commit messages.**
6.  **Push your branch** to your fork: `git push origin feature/your-feature-name`.
7.  **Open a Pull Request** against the `main` branch of `llamasearchai/OpenReranker`.
8.  Provide a clear description of your changes in the PR.

Please refer to `CONTRIBUTING.md` (if available) for more detailed guidelines.

## License

OpenReranker is licensed under the [MIT License](LICENSE).