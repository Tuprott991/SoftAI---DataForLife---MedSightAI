"""
Common schemas and utilities
"""
from pydantic import BaseModel
from typing import Optional, Any


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    detail: Optional[Any] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    s3: str
    milvus: str


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = 1
    page_size: int = 20


class ImageUploadResponse(BaseModel):
    """Image upload response"""
    file_path: str
    file_size: int
    content_type: str


class SimilaritySearchRequest(BaseModel):
    """Similarity search request"""
    case_id: Optional[str] = None
    image_path: Optional[str] = None
    text_description: Optional[str] = None
    top_k: int = 5


class SimilaritySearchResponse(BaseModel):
    """Similarity search response"""
    similar_cases: list[str]
    similarity_scores: list[float]
    case_details: Optional[list[dict]] = None
