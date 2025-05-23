from typing import Dict, List, Optional, Set
import json

from pydantic_settings import BaseSettings
from pydantic import field_validator, Field, ConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    # API Configuration
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]

    # Authentication
    AUTH_ENABLED: bool = True
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week

    # Redis Configuration (for rate limiting and caching)
    REDIS_URL: Optional[str] = None
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour

    # Rate Limiting
    RATE_LIMIT_QPS: int = 100  # Queries per second
    RATE_LIMIT_RPM: int = 600  # Requests per minute
    RATE_LIMIT_TPM: int = 1_000_000  # Tokens per minute for free tier

    # Token Counting
    COUNT_TOKENS: bool = True

    # Model Configuration
    DEFAULT_RERANKER_MODEL: str = "jina/reranker-v2"
    CODE_RERANKER_MODEL: str = "jina/reranker-code-v2"
    TABLE_RERANKER_MODEL: str = "jina/reranker-table-v2"

    # Models for different tiers
    TIER_MODELS: Dict[str, str] = {
        "free": "jina/reranker-v1",
        "pro": "jina/reranker-v2",
        "enterprise": "jina/reranker-v2",
    }

    # MLX Configuration
    USE_MLX: bool = True
    MLX_DEVICE: str = "gpu"  # Options: "gpu", "cpu"

    # Batch Configuration
    MAX_BATCH_SIZE: int = 32
    MAX_DOCUMENTS_PER_QUERY: int = 1000
    MAX_QUERY_LENGTH: int = 1024
    MAX_DOCUMENT_LENGTH: int = 8192

    # Performance Configuration
    NUM_WORKERS: int = 4

    # Monitoring
    ENABLE_MONITORING: bool = True

    # DSPy Configuration
    DSPY_DEFAULT_LM: str = "gpt-3.5-turbo"

    # LangChain Configuration
    LANGCHAIN_CACHE_ENABLED: bool = True

    # Language Support
    SUPPORTED_LANGUAGES: Set[str] = {
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "it",
    }


settings = Settings()
