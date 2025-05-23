import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from open_reranker.api.router import router as api_router
from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging
from open_reranker.core.monitoring import get_metrics, setup_metrics

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application.
    Used for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Open-Reranker API service")
    setup_metrics()
    yield
    # Shutdown
    logger.info("Shutting down Open-Reranker API service")


app = FastAPI(
    title="Open-Reranker API",
    description="Open-source reranker for maximizing search relevancy with DSPy and LangChain integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()

    # Track metrics before processing
    get_metrics().requests_in_progress.inc()

    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Track metrics after processing
    get_metrics().requests_in_progress.dec()
    get_metrics().request_duration.labels(
        method=request.method, endpoint=request.url.path
    ).observe(process_time)
    get_metrics().requests_total.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    # Log request details
    logger.info(
        f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s",
        extra={
            "path": request.url.path,
            "method": request.method,
            "duration": process_time,
            "status_code": response.status_code,
        },
    )

    return response


# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception for request {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
        },
    )

    if settings.ENABLE_MONITORING:
        metrics = get_metrics()
        metrics.exceptions_total.labels(type=type(exc).__name__).inc()

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
        },
    )


# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "open-reranker",
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics().export()


if __name__ == "__main__":
    uvicorn.run(
        "open_reranker.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
