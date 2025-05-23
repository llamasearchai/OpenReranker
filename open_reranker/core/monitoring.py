import functools
import time
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    REGISTRY,
)


class Metrics:
    """
    Prometheus metrics for monitoring.
    """

    def __init__(self):
        """Initialize metrics with a new registry."""
        self.registry = CollectorRegistry()

        # Request metrics
        self.requests_total = Counter(
            "open_reranker_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.requests_in_progress = Gauge(
            "open_reranker_requests_in_progress",
            "Number of requests in progress",
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "open_reranker_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=(
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ),
            registry=self.registry,
        )

        # Exception metrics
        self.exceptions_total = Counter(
            "open_reranker_exceptions_total",
            "Total number of exceptions",
            ["type"],
            registry=self.registry,
        )

        # Reranker metrics
        self.rerank_count = Counter(
            "open_reranker_rerank_count",
            "Number of rerank operations",
            ["model", "query_type"],
            registry=self.registry,
        )

        self.rerank_duration = Histogram(
            "open_reranker_rerank_duration_seconds",
            "Rerank operation duration in seconds",
            ["model"],
            buckets=(
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ),
            registry=self.registry,
        )

        # Model metrics
        self.model_load_duration = Histogram(
            "open_reranker_model_load_duration_seconds",
            "Model load duration in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0),
            registry=self.registry,
        )

        # Integration metrics
        self.integration_calls = Counter(
            "open_reranker_integration_calls_total",
            "Number of integration calls",
            ["integration", "operation"],
            registry=self.registry,
        )

    def export(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")


_METRICS: Optional[Metrics] = None


def setup_metrics() -> Metrics:
    """
    Set up metrics for monitoring.

    Returns:
        The Metrics instance.
    """
    global _METRICS

    if _METRICS is None:
        _METRICS = Metrics()

    return _METRICS


def get_metrics() -> Metrics:
    """
    Get the metrics instance.

    Returns:
        The Metrics instance.
    """
    global _METRICS

    if _METRICS is None:
        _METRICS = setup_metrics()

    return _METRICS


def track_time(name: str, metric_name: str):
    """
    Decorator to track function execution time.

    Args:
        name: The name of the operation (for logging)
        metric_name: The name of the metric to update
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Get metrics and record duration
            metrics = get_metrics()
            if hasattr(metrics, metric_name):
                metric = getattr(metrics, metric_name)

                # Extract model name if available in kwargs or first arg
                model = kwargs.get("model", "default")
                if model == "default" and hasattr(args[0], "model_name"):
                    model = args[0].model_name

                # Record the metric
                metric.labels(model=str(model)).observe(duration)

            return result

        return wrapper

    return decorator
