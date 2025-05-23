from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from open_reranker.api.models import (
    BatchRerankerRequest,
    BatchRerankerResponse,
    RerankerRequest,
    RerankerResponse,
)
from open_reranker.core.auth import get_current_user_optional, require_tier
from open_reranker.core.cache import cache_manager
from open_reranker.core.logging import setup_logging
from open_reranker.core.monitoring import get_metrics
from open_reranker.core.rate_limiting import check_rate_limits
from open_reranker.services.reranker_service import RerankerService

logger = setup_logging()
router = APIRouter()


def get_reranker_service():
    """Dependency to get a configured RerankerService instance."""
    return RerankerService()


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """Dependency to get optional current user."""
    return await get_current_user_optional(request)


@router.post(
    "/rerank",
    response_model=RerankerResponse,
    summary="Rerank documents based on query relevance",
    tags=["Reranking"],
)
async def rerank_documents(
    request: RerankerRequest,
    current_request: Request,
    reranker_service: RerankerService = Depends(get_reranker_service),
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Rerank documents based on their relevance to the query.

    - **query**: The search query
    - **documents**: List of documents to rerank
    - **model**: Reranker model to use
    - **top_k**: Number of top documents to return
    - **include_scores**: Whether to include scores in the response
    """
    try:
        # Apply rate limiting
        await check_rate_limits(
            current_request,
            user,
            token_count=len(request.query.split())
            + sum(len(doc.text.split()) for doc in request.documents),
        )

        # Check cache first
        document_texts = [doc.text for doc in request.documents]
        cached_result = await cache_manager.get_rerank_result(
            query=request.query,
            documents=document_texts,
            model=request.model,
            top_k=request.top_k,
            include_scores=request.include_scores,
            use_mlx=request.use_mlx,
        )

        if cached_result is not None:
            logger.info("Serving rerank result from cache")
            reranked = cached_result
            reranker_service.last_rerank_time = 0.001  # Cached response time
        else:
            # Rerank documents
            reranked = await reranker_service.rerank(
                query=request.query,
                documents=document_texts,
                model=request.model,
                top_k=request.top_k,
                include_scores=request.include_scores,
                use_mlx=request.use_mlx,
            )

            # Cache the result
            await cache_manager.set_rerank_result(
                query=request.query,
                documents=document_texts,
                model=request.model,
                result=reranked,
                top_k=request.top_k,
                include_scores=request.include_scores,
                use_mlx=request.use_mlx,
            )

        # Format response
        results = []
        for idx, score in reranked:
            doc = request.documents[idx]
            results.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": score if request.include_scores else None,
                    "original_position": idx,
                    "metadata": doc.metadata,
                }
            )

        # Create response
        response = {
            "results": results,
            "query": request.query,
            "timing": {
                "reranking_time": reranker_service.last_rerank_time,
            },
            "metadata": {
                "model": str(request.model),
                "input_document_count": len(request.documents),
                "cached": cached_result is not None,
            },
        }

        return response

    except Exception as e:
        # Log error
        logger.error(
            f"Error in rerank endpoint: {str(e)}",
            extra={
                "query": request.query,
                "document_count": len(request.documents),
                "model": str(request.model),
                "error": str(e),
            },
            exc_info=True,
        )

        # Update metrics
        metrics = get_metrics()
        metrics.exceptions_total.labels(type=type(e).__name__).inc()

        # Raise exception
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}",
        )


@router.post(
    "/rerank/batch",
    response_model=BatchRerankerResponse,
    summary="Batch rerank multiple queries",
    tags=["Reranking"],
)
async def batch_rerank_documents(
    request: BatchRerankerRequest,
    current_request: Request,
    reranker_service: RerankerService = Depends(get_reranker_service),
    user: Dict[str, Any] = Depends(require_tier("pro")),
):
    """
    Batch rerank multiple queries (Pro tier and above).

    - **queries**: List of search queries
    - **documents**: List of document lists for each query
    - **model**: Reranker model to use
    - **top_k**: Number of top documents to return for each query
    - **include_scores**: Whether to include scores in the response
    """
    try:
        # Calculate total tokens for rate limiting
        total_tokens = sum(len(query.split()) for query in request.queries)
        for doc_list in request.documents:
            total_tokens += sum(len(doc.text.split()) for doc in doc_list)

        # Apply rate limiting
        await check_rate_limits(current_request, user, token_count=total_tokens)

        results = []
        total_time = 0.0

        for query, documents in zip(request.queries, request.documents):
            # Process each query-document pair
            document_texts = [doc.text for doc in documents]

            reranked = await reranker_service.rerank(
                query=query,
                documents=document_texts,
                model=request.model,
                top_k=request.top_k,
                include_scores=request.include_scores,
                use_mlx=request.use_mlx,
            )

            # Format results for this query
            query_results = []
            for idx, score in reranked:
                doc = documents[idx]
                query_results.append(
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "score": score if request.include_scores else None,
                        "original_position": idx,
                        "metadata": doc.metadata,
                    }
                )

            results.append(
                {
                    "results": query_results,
                    "query": query,
                    "timing": {
                        "reranking_time": reranker_service.last_rerank_time,
                    },
                    "metadata": {
                        "model": str(request.model),
                        "input_document_count": len(documents),
                    },
                }
            )

            total_time += reranker_service.last_rerank_time

        return {
            "results": results,
            "timing": {
                "total_time": total_time,
                "average_time": (
                    total_time / len(request.queries) if request.queries else 0
                ),
            },
            "metadata": {
                "total_queries": len(request.queries),
                "model": str(request.model),
            },
        }

    except Exception as e:
        logger.error(f"Batch rerank error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch reranking failed: {str(e)}",
        )


@router.post(
    "/integrations/dspy",
    summary="DSPy integration endpoint",
    tags=["Integrations"],
)
async def dspy_integration(
    query: str = Body(..., embed=True),
    documents: List[str] = Body(..., embed=True),
    model: str = Body("default", embed=True),
    top_k: int = Body(10, embed=True),
    current_request: Request = None,
    reranker_service: RerankerService = Depends(get_reranker_service),
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    DSPy integration endpoint for reranking.

    - **query**: The search query
    - **documents**: List of document texts to rerank
    - **model**: Reranker model to use
    - **top_k**: Number of top documents to return
    """
    try:
        # Apply rate limiting
        token_count = len(query.split()) + sum(len(doc.split()) for doc in documents)
        await check_rate_limits(current_request, user, token_count=token_count)

        # Get metrics
        metrics = get_metrics()
        metrics.integration_calls.labels(integration="dspy", operation="rerank").inc()

        # Rerank documents
        reranked = await reranker_service.rerank(
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
            include_scores=True,
        )

        # Format as expected by DSPy
        results = [(documents[idx], float(score)) for idx, score in reranked]

        return {"results": results, "query": query, "model": model}

    except Exception as e:
        logger.error(f"DSPy integration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"DSPy integration failed: {str(e)}",
        )


@router.post(
    "/integrations/langchain",
    summary="LangChain integration endpoint",
    tags=["Integrations"],
)
async def langchain_integration(
    query: str = Body(..., embed=True),
    documents: List[Dict[str, Any]] = Body(..., embed=True),
    model: str = Body("default", embed=True),
    top_k: int = Body(10, embed=True),
    current_request: Request = None,
    reranker_service: RerankerService = Depends(get_reranker_service),
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    LangChain integration endpoint for reranking.

    - **query**: The search query
    - **documents**: List of document dicts to rerank
    - **model**: Reranker model to use
    - **top_k**: Number of top documents to return
    """
    try:
        # Apply rate limiting
        document_texts = [doc.get("page_content", "") for doc in documents]
        token_count = len(query.split()) + sum(
            len(text.split()) for text in document_texts
        )
        await check_rate_limits(current_request, user, token_count=token_count)

        # Get metrics
        metrics = get_metrics()
        metrics.integration_calls.labels(
            integration="langchain", operation="rerank"
        ).inc()

        # Rerank documents
        reranked = await reranker_service.rerank(
            query=query,
            documents=document_texts,
            model=model,
            top_k=top_k,
            include_scores=True,
        )

        # Format results in LangChain format
        results = []
        for idx, score in reranked:
            doc = documents[idx].copy()

            # Add score to metadata
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"]["reranker_score"] = float(score)

            results.append(doc)

        return {"results": results, "query": query, "model": model}

    except Exception as e:
        logger.error(f"LangChain integration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"LangChain integration failed: {str(e)}",
        )


@router.delete(
    "/cache",
    summary="Clear cache",
    tags=["Admin"],
)
async def clear_cache(user: Dict[str, Any] = Depends(require_tier("enterprise"))):
    """
    Clear all cached results (Enterprise tier only).
    """
    try:
        await cache_manager.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}",
        )
