"""
Pydantic schemas for Case
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class CaseBase(BaseModel):
    """Base case schema"""
    patient_id: UUID = Field(..., description="Patient ID")


class CaseCreate(CaseBase):
    """Schema for creating a case"""
    pass


class CaseUpdate(BaseModel):
    """Schema for updating a case"""
    processed_img_path: Optional[str] = None
    similar_cases: Optional[List[str]] = None
    similarity_scores: Optional[List[float]] = None


class CaseResponse(CaseBase):
    """Schema for case response"""
    id: UUID
    image_path: str
    processed_img_path: Optional[str]
    timestamp: datetime
    similar_cases: Optional[List[str]]
    similarity_scores: Optional[List[float]]
    
    class Config:
        from_attributes = True


class CaseWithPatientResponse(CaseResponse):
    """Schema for case response with patient info"""
    patient_name: str
    patient_age: Optional[int]
    patient_gender: Optional[str]


class CaseListResponse(BaseModel):
    """Schema for paginated case list"""
    total: int
    page: int
    page_size: int
    cases: List[CaseResponse]
