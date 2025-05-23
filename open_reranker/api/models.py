from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class Document(BaseModel):
    """Model representing a document to be reranked."""

    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata for the document"
    )


class RerankerRequest(BaseModel):
    """Request model for text reranking."""

    query: str = Field(..., description="The search query")
    documents: List[Document] = Field(..., description="Documents to rerank")
    model: str = Field(default="default", description="Reranker model to use")
    top_k: Optional[int] = Field(
        default=None, description="Number of top documents to return"
    )
    include_scores: bool = Field(
        default=True, description="Whether to include scores in the response"
    )
    use_mlx: bool = Field(
        default=True, description="Whether to use MLX acceleration if available"
    )


class RankedDocument(BaseModel):
    """Model representing a reranked document."""

    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Text content of the document")
    score: Optional[float] = Field(
        default=None, description="Relevance score (higher is better)"
    )
    original_position: int = Field(
        ..., description="Original position in the input documents"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata for the document"
    )


class RerankerResponse(BaseModel):
    """Response model for text reranking."""

    results: List[RankedDocument] = Field(..., description="Reranked documents")
    query: str = Field(..., description="The original search query")
    timing: Dict[str, float] = Field(
        default_factory=dict, description="Timing information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class BatchRerankerRequest(BaseModel):
    """Request model for batch reranking."""

    queries: List[str] = Field(..., description="List of search queries")
    documents: List[List[Document]] = Field(
        ..., description="List of document lists for each query"
    )
    model: str = Field(default="default", description="Reranker model to use")
    top_k: Optional[int] = Field(
        default=None, description="Number of top documents to return for each query"
    )
    include_scores: bool = Field(
        default=True, description="Whether to include scores in the response"
    )
    use_mlx: bool = Field(
        default=True, description="Whether to use MLX acceleration if available"
    )


class BatchRerankerResponse(BaseModel):
    """Response model for batch reranking."""

    results: List[RerankerResponse] = Field(
        ..., description="Reranking results for each query"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict, description="Timing information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
