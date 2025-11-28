"""
Education Mode API endpoints
"""
from uuid import UUID
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import chat_session as crud_chat_session, chat_message as crud_chat_message
from app.schemas import (
    ChatSessionCreate, ChatSessionResponse,
    ChatMessageCreate, ChatMessageResponse,
    ChatHistoryResponse, StudentSubmission, StudentScoreResponse,
    MessageResponse
)
from app.services import medgemma_service, ai_model_service

router = APIRouter()


@router.get("/practice-cases")
async def get_practice_cases(
    disease_type: str = Query(None),
    difficulty: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get practice cases for education mode
    Filtered by disease type and difficulty
    
    TODO: Implement case selection logic
    - Query cases from database
    - Filter by disease type
    - Optionally filter by difficulty level
    - Return unlabeled images for practice
    """
    raise NotImplementedError("Implement practice case selection")


@router.post("/submit-answer", response_model=StudentScoreResponse)
async def submit_student_answer(
    submission: StudentSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit student's diagnosis and bounding boxes
    Calculate score and provide feedback
    
    TODO: Implement scoring algorithm
    1. Compare student's bounding boxes with ground truth (IoU metric)
    2. Compare diagnosis with correct answer
    3. Calculate overall score
    4. Generate detailed feedback using MedGemma
    """
    # Get session
    session = crud_chat_session.get(db, submission.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Get ground truth for the practice case
    # ground_truth = get_ground_truth(session.image_path)
    
    # TODO: Calculate bounding box accuracy (IoU)
    # bbox_accuracy = calculate_bbox_accuracy(submission.bounding_boxes, ground_truth['bboxes'])
    
    # TODO: Calculate diagnosis accuracy
    # diagnosis_accuracy = calculate_diagnosis_accuracy(submission.diagnosis, ground_truth['diagnosis'])
    
    # TODO: Generate feedback
    # feedback = medgemma_service.generate_feedback(
    #     student_answer={
    #         'diagnosis': submission.diagnosis,
    #         'bounding_boxes': submission.bounding_boxes
    #     },
    #     correct_answer=ground_truth
    # )
    
    # TODO: Generate comparison heatmap
    # heatmap_path = generate_comparison_heatmap(...)
    
    raise NotImplementedError("Implement student scoring and feedback")


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_chat_session(
    session_in: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new chat session for student learning"""
    session_data = session_in.model_dump()
    session = crud_chat_session.create(db, obj_in=session_data)
    return session


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    user_id: UUID = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all chat sessions for a user"""
    skip = (page - 1) * page_size
    sessions = crud_chat_session.get_by_user(db, user_id=user_id, skip=skip, limit=page_size)
    return sessions


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Get chat history for a session"""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = crud_chat_message.get_by_session(db, session_id=session_id)
    
    return {
        "session": session,
        "messages": messages
    }


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    session_id: UUID,
    message: str,
    db: Session = Depends(get_db)
):
    """
    Send a message in chat session and get AI response
    
    TODO: Call medgemma_service.generate_chat_response()
    """
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save user message
    user_message_data = {
        "session_id": session_id,
        "from_role": "user",
        "message": message
    }
    user_message = crud_chat_message.create(db, obj_in=user_message_data)
    
    # TODO: Get conversation history
    # history = crud_chat_message.get_by_session(db, session_id=session_id)
    
    # TODO: Generate AI response
    # ai_response = medgemma_service.generate_chat_response(
    #     conversation_history=history,
    #     student_query=message,
    #     image_context={"image_path": session.image_path}
    # )
    
    # Save AI message
    # ai_message_data = {
    #     "session_id": session_id,
    #     "from_role": "assistant",
    #     "message": ai_response
    # }
    # ai_message = crud_chat_message.create(db, obj_in=ai_message_data)
    
    raise NotImplementedError("Connect to MedGemma chat service")


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def delete_chat_session(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a chat session"""
    session = crud_chat_session.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    crud_chat_session.delete(db, id=session_id)
    return {"message": "Chat session deleted successfully"}
