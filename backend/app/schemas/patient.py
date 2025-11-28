"""
Pydantic schemas for Patient
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class PatientBase(BaseModel):
    """Base patient schema"""
    name: str = Field(..., description="Patient name")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    history: Optional[Dict[str, Any]] = Field(None, description="Medical history including symptoms, past conditions, test results")


class PatientCreate(PatientBase):
    """Schema for creating a patient"""
    pass


class PatientUpdate(BaseModel):
    """Schema for updating a patient"""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    history: Optional[Dict[str, Any]] = None


class PatientResponse(PatientBase):
    """Schema for patient response"""
    id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True


class PatientListResponse(BaseModel):
    """Schema for paginated patient list"""
    total: int
    page: int
    page_size: int
    patients: list[PatientResponse]
