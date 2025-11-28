"""
Pydantic schemas for Chat (Education Mode)
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from uuid import UUID


class ChatSessionCreate(BaseModel):
    """Schema for creating a chat session"""
    user_id: UUID = Field(..., description="Student user ID")
    image_path: Optional[str] = Field(None, description="Practice image path")


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    id: UUID
    user_id: UUID
    image_path: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message"""
    session_id: UUID
    from_role: Literal["assistant", "user"]
    message: str
    submitted_image: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    id: UUID
    session_id: UUID
    from_role: str
    message: str
    submitted_image: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Schema for chat history"""
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]


class StudentSubmission(BaseModel):
    """Schema for student diagnosis submission"""
    session_id: UUID
    diagnosis: str
    bounding_boxes: List[dict]
    confidence: Optional[float] = None


class StudentScoreResponse(BaseModel):
    """Schema for student scoring response"""
    total_score: float
    bbox_accuracy: float
    diagnosis_accuracy: float
    explanation: str
    correct_answer: dict
    heatmap_path: Optional[str] = None
