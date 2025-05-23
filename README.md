# Open-Reranker

An open-source reranker service for maximizing search relevancy with DSPy and LangChain integration. Built with cutting-edge models and optimized for performance with MLX acceleration on Apple Silicon.

## Features

- **High-Performance Reranking**: Uses state-of-the-art cross-encoder models for maximizing search relevance.
- **Framework Integrations**: Seamlessly integrates with DSPy and LangChain (now with async support and direct service usage).
- **Specialized Rerankers**: Dedicated models for text, code, and tabular data (code/table are future enhancements).
- **Performance Optimizations**: MLX acceleration for Apple Silicon.
- **REST API**: Easy-to-use asynchronous API for all reranking operations.
- **Production-Ready**:
    - Comprehensive logging (JSON format) and monitoring (Prometheus).
    - JWT-based Authentication.
    - Tier-based Rate Limiting (QPS, RPM, TPM) with Redis or in-memory fallback.
    - Response Caching (model scores and full rerank responses) with Redis or in-memory fallback.
    - Scalable with configurable worker counts.

## Installation

### Using pip

```bash
pip install open-reranker
```

### From source

```bash
git clone https://github.com/your-username/open-reranker.git # Replace with your actual repo URL
cd open-reranker

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,integrations,mlx]" # Install all optional dependencies
# or pip install -e . for core dependencies only
```

## Quick Start

### 1. Setup Environment Variables

Copy the `.env.example` file to `.env` and customize it:

```bash
cp .env.example .env
# Edit .env with your settings (e.g., SECRET_KEY, REDIS_URL if used)
```

### 2. Start the Server

```bash
# Using Uvicorn (for development or single instance)
# The number of workers can be set via OPEN_RERANKER_NUM_WORKERS in .env or --workers flag
uvicorn open_reranker.main:app --host 0.0.0.0 --port 8000 --workers 4 

# For development with auto-reload (typically 1 worker):
# uvicorn open_reranker.main:app --reload

# For production, consider using Gunicorn with Uvicorn workers:
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker open_reranker.main:app -b 0.0.0.0:8000
```

### 3. Using the API

**Example: Reranking Documents**

```python
import requests
import json

# Define your API endpoint
API_URL = "http://localhost:8000/api/v1/rerank"

# Define your query and documents
query = "What is the function of chlorophyll in plants?"
documents = [
    {"id": "doc1", "text": "Chlorophyll is a green pigment found in plants that absorbs light energy for photosynthesis."},
    {"id": "doc2", "text": "Plants use photosynthesis to convert carbon dioxide and water into glucose and oxygen."},
    {"id": "doc3", "text": "The Earth orbits around the Sun at an average distance of 149.6 million kilometers."}
]

# Create request payload
payload = {
    "query": query,
    "documents": documents,
    "model": "default", # or specify a model like "jinaai/jina-reranker-v2-base-en"
    "top_k": 2,
    "include_scores": True
}

# If authentication is enabled (OPEN_RERANKER_AUTH_ENABLED=True in .env)
# You'll need to obtain a JWT token first (e.g., via a /token endpoint if implemented)
# and include it in the headers.
# HEADERS = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
# response = requests.post(API_URL, json=payload, headers=HEADERS)

# Assuming auth is disabled or token is handled for this example:
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    results = response.json()
    print(f"Query: {results['query']}")
    print(f"Processing time: {results['timing']['reranking_time']:.3f} seconds")
    print(f"Cached: {results['metadata'].get('cached', False)}")
    
    print("\nRanked Results:")
    for i, doc in enumerate(results["results"]):
        print(f"{i+1}. Score: {doc['score']:.4f} - ID: {doc['id']} - {doc['text'][:100]}...")
elif response.status_code == 401:
    print(f"Authentication Error: {response.status_code} - {response.text}")
elif response.status_code == 429:
    print(f"Rate Limit Error: {response.status_code} - {response.text}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Using with LangChain

The `LangChainReranker` integration now uses the `RerankerService` directly and supports asynchronous operations.

```python
import asyncio
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

# Example of a simple custom retriever for LangChain for demonstration
class MySimpleRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # This is the synchronous version
        # In a real scenario, you might have an async version or use asyncio.run for a sync wrapper
        print("(MySimpleRetriever._get_relevant_documents called)")
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        print("(MySimpleRetriever._aget_relevant_documents called)")
        # Simulate async work if needed
        await asyncio.sleep(0.01)
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]


from open_reranker.integrations.langchain import LangChainReranker

# Create a base retriever (replace with your actual retriever)
base_docs = [
    Document(page_content="Document 1 about science and chlorophyll."),
    Document(page_content="Document 2 about history."),
    Document(page_content="Document 3 about biology and plants, focusing on chlorophyll.")
]
base_retriever = MySimpleRetriever(docs=base_docs)


# Create a reranker on top of the base retriever
# LangChainReranker uses RerankerService directly (not an HTTP endpoint)
reranker = LangChainReranker(
    base_retriever=base_retriever,
    model="default", # Uses the default model configured in OpenReranker service
    top_k=2
)

async def main_langchain():
    # Query with reranking (asynchronously)
    print("\n--- LangChain Async Test ---")
    results_async = await reranker.aget_relevant_documents("Tell me about science and plants")
    print("Async Results:")
    for doc in results_async:
        print(f"Score: {doc.metadata.get('reranker_score', 0.0):.4f} - {doc.page_content}")

    # Synchronous usage (calls async version internally with asyncio.run)
    print("\n--- LangChain Sync Test ---")
    results_sync = reranker.get_relevant_documents("Tell me about science and plants")
    print("Sync Results (via async wrapper):")
    for doc in results_sync:
        print(f"Score: {doc.metadata.get('reranker_score', 0.0):.4f} - {doc.page_content}")

if __name__ == "__main__":
    # To run this example, ensure OpenReranker is installed and its dependencies are available.
    # You might need to run this from the root of the OpenReranker project or ensure PYTHONPATH is set.
    # Also, ensure the RerankerService can load models (e.g., transformers are installed).
    # asyncio.run(main_langchain())
    print("LangChain example: Run with `asyncio.run(main_langchain())` in a suitable environment.")
```

### Using with DSPy

The `DSPyReranker` (formerly `OpenRerankerRM`) integration now uses the `RerankerService` directly and supports asynchronous operations.

```python
import asyncio
import dspy
from typing import List, Union, Any, Optional

# Example of a simple custom RM for DSPy for demonstration
class MySimpleRM(dspy.Retrieve):
    def __init__(self, docs: List[str], k: int = 3):
        super().__init__(k=k)
        self.docs = [dspy.Passage(long_text=doc) for doc in docs]

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> List[dspy.Passage]:
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        results = []
        for query in queries:
            # Simple keyword match for demonstration
            matched_docs = [doc for doc in self.docs if query.lower() in doc.long_text.lower()]
            results.extend(dspy.Passage.sort_passages_by_scores(matched_docs)[:k]) # DSPy expects sorted
        return results

from open_reranker.integrations.dspy.reranker import DSPyReranker

# Dummy documents for the base retriever
sample_passages_text = [
    "Chlorophyll is essential for photosynthesis in plants.",
    "The Roman Empire was vast and influential.",
    "Quantum physics explores the nature of reality at the smallest scales.",
    "Plants absorb sunlight using chlorophyll, a key pigment."
]

# Create a base retrieval module
base_rm = MySimpleRM(docs=sample_passages_text, k=10)

# Create an OpenReranker retrieval module
# DSPyReranker uses RerankerService directly
reranker_rm = DSPyReranker(
    base_retriever=base_rm,
    model="default", 
    top_k=2
)

# Example DSPy Module using the reranker
class RAGWithDSPyReranker(dspy.Module):
    def __init__(self):
        super().__init__()
        # DSPyReranker acts as a retriever itself, taking a base_retriever
        self.retrieve_and_rerank = reranker_rm
        # Example of a predictor that would use the retrieved & reranked context
        # self.generate_answer = dspy.Predict('context, question -> answer') 

    def forward(self, query: str) -> Any:
        # DSPyReranker's forward method returns List[Tuple[str, float]] (text, score)
        # We need to adapt this to dspy.Passage objects for use in other DSPy modules.
        # The DSPyReranker itself now returns List[dspy.Passage] when called directly or via __call__ (sync)
        # if it wraps a base_retriever that returns dspy.Passage objects.
        # For this example, let's assume we get (text, score) and convert.

        # The DSPyReranker is a Retriever, so its forward/call returns List[dspy.Passage]
        # if the base_retriever returns dspy.Passage. MySimpleRM returns dspy.Passage.
        context = self.retrieve_and_rerank(query) # This calls DSPyReranker.__call__ (sync)
                                                 # which wraps its async forward.
        
        # For demonstration, we'll just return the context.
        # In a real RAG, you would pass this to a dspy.Predict or dspy.ChainOfThought module.
        # e.g., prediction = self.generate_answer(context=context, question=query)
        return context 

async def main_dspy():
    print("\n--- DSPy Async Test (Illustrative) ---")
    # DSPy itself is primarily synchronous in its main execution flow for modules.
    # The DSPyReranker.forward is async, and __call__ wraps it.
    # To test the async nature, you'd typically call the .forward() method of the retriever directly.
    
    # Get initial passages from base retriever
    base_passages = base_rm.forward("What is chlorophyll in plants?")
    print(f"Base RM found {len(base_passages)} passages.")

    # Rerank these using the async forward method of DSPyReranker
    # Note: DSPyReranker.forward expects a query string and documents as list of strings or dspy.Passage
    # and returns List[Tuple[str, float]]. We need to adapt.

    # The `reranker_rm` when called directly (or its `forward` method) will use its `base_retriever`
    # This means the following call will first call base_rm.forward, then rerank.
    async_reranked_results_with_scores = await reranker_rm.forward("What is chlorophyll in plants?")
    print("Async Reranked Results (text, score tuples from DSPyReranker.forward):")
    # reranker_rm.forward returns List[Tuple[Any, float]], where Any is the passage text
    for text, score in async_reranked_results_with_scores:
        print(f"Score: {score:.4f} - {text}")

    print("\n--- DSPy Sync Test (using RAG module) ---")
    # Configure DSPy (optional, if you have a global LM/RM setup)
    # For this standalone example, it's not strictly needed as RAGWithDSPyReranker uses specific modules.
    # try:
    #     dspy.settings.configure(lm=None, rm=None) # Dummy config
    # except AttributeError: # Handle older DSPy versions
    #     pass 

    rag_program = RAGWithDSPyReranker()
    results = rag_program(query="What is chlorophyll in plants?") # Calls __call__ on reranker_rm
    
    print("Sync RAG Results (dspy.Passage objects):")
    for passage in results:
        # The score should ideally be part of the dspy.Passage object if set by the reranker
        # The current DSPyReranker.forward returns (text, score), so the RAG module needs to construct dspy.Passage with score.
        # Let's assume the score is in passage.score or passage.metadata['score']
        score = passage.score if hasattr(passage, 'score') and passage.score is not None else passage.long_text.split(" (Score: ")[-1].replace(")","") if " (Score: " in passage.long_text else 0.0
        try:
            score = float(score)
        except ValueError:
            score = 0.0 # Default if score parsing fails
        print(f"Score: {score:.4f} - {passage.long_text}")

if __name__ == "__main__":
    # To run this example, ensure OpenReranker & DSPy are installed and dependencies are available.
    # You might need to run this from the root of the OpenReranker project or ensure PYTHONPATH is set.
    # Also, ensure the RerankerService can load models.
    # asyncio.run(main_dspy())
    print("DSPy example: Run with `asyncio.run(main_dspy())` in a suitable environment.")

```

## API Endpoints

-   **`POST /api/v1/rerank`**: Reranks a list of documents for a given query.
    -   Supports caching (full response) and rate limiting. Asynchronous.
-   **`POST /api/v1/rerank/batch`**: Reranks multiple sets of documents for multiple queries in a single request.
    -   Requires "pro" tier or higher by default (configurable via JWT tier claim).
    -   Supports rate limiting. Asynchronous.
-   **`POST /api/v1/rerank/code`**: (Future Enhancement) Reranks code snippets.
-   **`POST /api/v1/rerank/table`**: (Future Enhancement) Reranks tabular data.
-   **`POST /api/v1/integrations/dspy`**: HTTP endpoint for DSPy integration (if not using `DSPyReranker` class directly).
    -   The `DSPyReranker` class is designed to use the `RerankerService` directly (preferred method).
    -   Supports rate limiting. Asynchronous.
-   **`POST /api/v1/integrations/langchain`**: HTTP endpoint for LangChain integration (if not using `LangChainReranker` class directly).
    -   The `LangChainReranker` class uses the `RerankerService` directly (preferred method).
    -   Supports rate limiting. Asynchronous.
-   **`GET /`**: Health check.
-   **`GET /metrics`**: Prometheus metrics endpoint.
-   **`DELETE /api/v1/cache`**: Clears all cached results (model scores and rerank responses).
    -   Requires "enterprise" tier or higher by default (configurable via JWT tier claim).

*(Authentication might be required for some endpoints if `OPEN_RERANKER_AUTH_ENABLED` is true.)*

## Configuration

The Open-Reranker service is configured via environment variables. Create a `.env` file in the project root (you can copy `.env.example`).

Key environment variables:

```bash
# --- General API Settings ---
OPEN_RERANKER_API_PREFIX="/api/v1"       # Base path for all API routes
OPEN_RERANKER_DEBUG=False                # Enable FastAPI debug mode (True/False)
OPEN_RERANKER_HOST="0.0.0.0"             # Host address to bind the server
OPEN_RERANKER_PORT=8000                  # Port for the server

# --- CORS (Cross-Origin Resource Sharing) ---
# Comma-separated list of allowed origins, or "*" for all.
# Example: OPEN_RERANKER_CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
OPEN_RERANKER_CORS_ORIGINS="*"

# --- Authentication --- 
OPEN_RERANKER_AUTH_ENABLED=True          # Enable JWT authentication (True/False)
# IMPORTANT: Change this in production! Used to sign JWTs.
OPEN_RERANKER_SECRET_KEY="your-super-secret-jwt-key-please-change-this" 
OPEN_RERANKER_ACCESS_TOKEN_EXPIRE_MINUTES=10080 # Default: 7 days (in minutes)
# Define user tiers for rate limiting and feature access (comma-separated)
OPEN_RERANKER_USER_TIERS="free,pro,enterprise" 
# Define required tier for /rerank/batch endpoint
OPEN_RERANKER_BATCH_ENDPOINT_TIER="pro"
# Define required tier for /cache DELETE endpoint
OPEN_RERANKER_CACHE_ENDPOINT_TIER="enterprise"

# --- Redis for Caching & Rate Limiting ---
# If not set, in-memory stores will be used (not suitable for multi-process/multi-instance).
# Example: OPEN_RERANKER_REDIS_URL="redis://localhost:6379/0"
OPEN_RERANKER_REDIS_URL=
OPEN_RERANKER_CACHE_ENABLED=True         # Enable response caching (True/False)
OPEN_RERANKER_CACHE_TTL=3600             # Default cache Time-To-Live in seconds (1 hour) for full responses
OPEN_RERANKER_MODEL_SCORE_CACHE_TTL=86400 # TTL for individual model scores (1 day)

# --- Rate Limiting --- 
# These are default limits for the "free" tier, applied per client ID (user or IP).
# Pro and Enterprise tiers typically have higher limits (scaled in code based on these defaults).
OPEN_RERANKER_RATE_LIMIT_QPS=100         # Queries Per Second
OPEN_RERANKER_RATE_LIMIT_RPM=6000        # Requests Per Minute (increased from 600)
OPEN_RERANKER_RATE_LIMIT_TPM=1000000     # Tokens Per Minute (if COUNT_TOKENS is True)
OPEN_RERANKER_COUNT_TOKENS=True          # Enable token counting for rate limiting (True/False)
# Scaling factors for pro and enterprise tiers (e.g., 2.0 means 2x the free tier limit)
OPEN_RERANKER_PRO_TIER_SCALE_FACTOR=2.0
OPEN_RERANKER_ENTERPRISE_TIER_SCALE_FACTOR=5.0

# --- Model Configuration ---
# Default models (Jina AI v2 Rerankers are good choices)
OPEN_RERANKER_DEFAULT_RERANKER_MODEL="jinaai/jina-reranker-v2-base-en" 
OPEN_RERANKER_CODE_RERANKER_MODEL="jinaai/jina-reranker-v2-base-code"  # For future code reranking
OPEN_RERANKER_TABLE_RERANKER_MODEL="jinaai/jina-reranker-v1-base- શું" # Example for table reranking (model to be chosen)

# --- Tier-Specific Models (Optional JSON string) ---
# Example: OPEN_RERANKER_TIER_MODELS='{"free": "jinaai/jina-reranker-v1-base-en", "pro": "jinaai/jina-reranker-v2-base-en", "enterprise": "jinaai/jina-reranker-v2-base-en"}'
OPEN_RERANKER_TIER_MODELS='{}'

# --- MLX Acceleration (for Apple Silicon) ---
OPEN_RERANKER_USE_MLX=True               # Attempt to use MLX if on Darwin (True/False)
OPEN_RERANKER_MLX_DEVICE="gpu"           # "gpu" or "cpu" for MLX

# --- Batch Processing & Performance ---
OPEN_RERANKER_MAX_BATCH_SIZE=32          # Max documents processed by model in one go (per internal batch)
OPEN_RERANKER_MAX_DOCUMENTS_PER_QUERY=1000 # Max documents accepted per query in /rerank API call
OPEN_RERANKER_MAX_QUERY_LENGTH=1024      # Max characters for query truncation
OPEN_RERANKER_MAX_DOCUMENT_LENGTH=8192   # Max characters for document truncation
# Number of Uvicorn workers (if not using a process manager like Gunicorn)
# Effective when running `uvicorn open_reranker.main:app --workers $OPEN_RERANKER_NUM_WORKERS`
# or if Gunicorn is not used.
OPEN_RERANKER_NUM_WORKERS=4              

# --- Monitoring ---
OPEN_RERANKER_ENABLE_MONITORING=True     # Enable Prometheus metrics endpoint (True/False)

# --- Integrations (Informational, as integrations use RerankerService directly now) ---
# OPEN_RERANKER_DSPY_DEFAULT_LM="gpt-3.5-turbo" 
# OPEN_RERANKER_LANGCHAIN_CACHE_ENABLED=True    

# --- Language Support (Informational, for future model selection logic) ---
# Comma-separated list of ISO language codes. Example: "en,zh,de,es,ru,ko,fr,ja,pt,it"
# OPEN_RERANKER_SUPPORTED_LANGUAGES="en"
```

### Setting up Redis

If you want to use Redis for distributed caching (model scores and full responses) and rate limiting (recommended for production or multi-worker setups):
1.  Install Redis: Follow instructions at [redis.io](https://redis.io/docs/getting-started/installation/).
2.  Start Redis server (usually `redis-server`).
3.  Set the `OPEN_RERANKER_REDIS_URL` in your `.env` file, e.g., `OPEN_RERANKER_REDIS_URL="redis://localhost:6379/0"`.

## Models

By default, Open-Reranker uses the following models (examples, can be configured):

-   **Text Reranking (Default)**: `jinaai/jina-reranker-v2-base-en`
-   **Code Reranking**: `jinaai/jina-reranker-v2-base-code` (for future `/rerank/code` endpoint)
-   **Table Reranking**: (e.g., a model specialized for tables, for future `/rerank/table` endpoint)

You can specify custom models by setting the appropriate environment variables (e.g., `OPEN_RERANKER_DEFAULT_RERANKER_MODEL`) or by passing the `model` parameter in API calls where supported. Tier-specific models can also be defined via `OPEN_RERANKER_TIER_MODELS`.

## Authentication and Authorization

If `OPEN_RERANKER_AUTH_ENABLED` is set to `True` (default):
-   Most API endpoints will require a valid JWT Bearer token in the `Authorization` header.
-   A mechanism to issue tokens (e.g., a `/token` endpoint with username/password login) would typically be part of a larger application incorporating Open-Reranker, or you might pre-generate tokens for service-to-service communication. For development, you can disable auth or use a known `SECRET_KEY` to generate tokens.
-   The JWT must contain a `sub` (subject/user ID) claim and a `tier` claim (e.g., "free", "pro", "enterprise" - configurable via `OPEN_RERANKER_USER_TIERS`).
-   Certain features like batch reranking (`/api/v1/rerank/batch`) and cache clearing (`/api/v1/cache`) require specific user tiers, configured by `OPEN_RERANKER_BATCH_ENDPOINT_TIER` and `OPEN_RERANKER_CACHE_ENDPOINT_TIER` respectively.

**Example JWT Payload:**
```json
{
  "sub": "user_id_or_service_name",
  "tier": "pro", 
  "exp": 1678886400 
}
```
The `OPEN_RERANKER_SECRET_KEY` in `.env` is crucial for signing and verifying these tokens.

## Rate Limiting

-   Limits are applied per client identifier (authenticated user ID from JWT `sub` claim, or fallback to IP address if auth is disabled or token is missing/invalid).
-   Different tiers ("free", "pro", "enterprise" - from JWT `tier` claim) have different rate limits. The scaling factors for pro/enterprise tiers are set by `OPEN_RERANKER_PRO_TIER_SCALE_FACTOR` and `OPEN_RERANKER_ENTERPRISE_TIER_SCALE_FACTOR` applied to the base "free" tier limits.
-   Supports QPS, RPM, and TPM (Tokens Per Minute, if `OPEN_RERANKER_COUNT_TOKENS=True`). Token counts include query and document lengths.
-   Uses Redis if `OPEN_RERANKER_REDIS_URL` is set, otherwise falls back to in-memory limiting (not suitable for distributed deployments with multiple workers/instances).

## Caching

-   **Full Response Caching**: Responses from the `/rerank` endpoint can be cached. TTL set by `OPEN_RERANKER_CACHE_TTL`.
-   **Model Score Caching**: Individual model scores (logits) for query-document pairs are cached by `RerankerService`. TTL set by `OPEN_RERANKER_MODEL_SCORE_CACHE_TTL`.
-   Uses Redis if `OPEN_RERANKER_REDIS_URL` is set and `OPEN_RERANKER_CACHE_ENABLED=True`. Otherwise, uses an in-memory cache for both types.
-   The cache can be cleared via the `DELETE /api/v1/cache` endpoint (requires tier specified in `OPEN_RERANKER_CACHE_ENDPOINT_TIER`).

## Development

### Setup Development Environment

```bash
# 1. Clone the repository (if you haven't already)
# git clone https://github.com/your-username/open-reranker.git 
# cd open-reranker

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/macOS
# .venv\Scripts\activate # Windows

# 3. Install dependencies (including development tools)
# Ensure you have pip version that supports pyproject.toml extras well
pip install --upgrade pip
pip install -e ".[dev,integrations,mlx]"

# 4. Setup pre-commit hooks (optional, but recommended)
pre-commit install

# 5. Create your .env file
cp .env.example .env
# Update .env with your local settings, especially OPEN_RERANKER_SECRET_KEY
# For testing auth, ensure OPEN_RERANKER_AUTH_ENABLED=True (default)

# 6. Run tests
pytest
```

### Running Linters and Type Checker

```bash
black .          # Formatter
isort .          # Import sorter
flake8 .        # Linter (combines pycodestyle, pyflakes, mccabe)
# bandit -r open_reranker -c pyproject.toml # Security linter (config in pyproject.toml)
mypy open_reranker # Type checker
```
Or use the Makefile:
```bash
make format     # Runs black and isort
make lint       # Runs flake8 and bandit
make typecheck  # Runs mypy
make test       # Runs pytest
make all-checks # Runs format, lint, typecheck, test
```

## Connecting with Other Programs/Services

Open-Reranker is designed as a microservice. Integrate it via:

1.  **HTTP API Calls**: (Recommended for decoupling)
    *   Other services make HTTP requests to Open-Reranker API endpoints (e.g., `/api/v1/rerank`).
    *   Use `requests` or `httpx` in other Python programs.
    *   Handle authentication (JWT Bearer tokens) if `OPEN_RERANKER_AUTH_ENABLED=True`.

2.  **Direct SDK/Class Usage**: (If OpenReranker is a library in the same Python environment)
    *   Import `RerankerService`, `LangChainReranker`, or `DSPyReranker` classes directly.
    *   This is suitable for monolithic applications or tightly coupled systems.
    *   Example:
        ```python
        # from open_reranker.services.reranker_service import RerankerService
        # service = RerankerService()
        # results = asyncio.run(service.rerank(query="..."), documents=["..."])
        ```

3.  **Message Queues**: (For advanced, asynchronous, decoupled workflows - not built-in)
    *   Set up RabbitMQ, Kafka, etc. Your services publish tasks, Open-Reranker consumes.

**Example: Calling Open-Reranker from another Python program via HTTP (Async with httpx):**

```python
# In your other_program.py
import httpx # Preferred for async
import asyncio

OPEN_RERANKER_API_URL = "http://localhost:8000/api/v1/rerank"

async def get_reranked_results(query: str, documents: list[dict], auth_token: str = None) -> list:
    payload = {
        "query": query,
        "documents": documents, # Each dict: {"id": "...", "text": "..."}
        "top_k": 5 
    }
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OPEN_RERANKER_API_URL, json=payload, headers=headers)
            response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
            return response.json().get("results", [])
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error calling Open-Reranker: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"Request Error calling Open-Reranker: {e}")
            return []

# Example usage:
# async def example_run():
#     my_query = "What is photosynthesis?"
#     extracted_docs = [
#         {"id": "docA", "text": "Photosynthesis is a process used by plants."}, 
#         {"id": "docB", "text": "The sun is a star."}
#     ]
#     # Assuming you have a way to get a token if auth is enabled
#     # my_jwt_token = get_my_jwt_token_function() 
#     my_jwt_token = None # For no-auth example
# 
#     reranked_docs = await get_reranked_results(my_query, extracted_docs, auth_token=my_jwt_token)
#     for doc in reranked_docs:
#         print(f"Reranked: {doc['id']} - Score: {doc['score']}")
#
# if __name__ == "__main__":
#    asyncio.run(example_run())
```

This setup ensures Open-Reranker can function as a standalone, specialized service that your other applications can leverage for its reranking capabilities, either via its HTTP API or by directly using its classes if appropriate for your architecture.

## License

MIT License