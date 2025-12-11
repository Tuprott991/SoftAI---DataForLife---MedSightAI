"""
Notification API endpoints
Send push notifications via FCM
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.fcm_service import fcm_service
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

router = APIRouter()

class SendNotificationRequest(BaseModel):
    user_id: str
    title: str
    body: str
    data: Optional[Dict[str, str]] = None

class SendMulticastRequest(BaseModel):
    user_ids: List[str]
    title: str
    body: str
    data: Optional[Dict[str, str]] = None

class SendTopicRequest(BaseModel):
    topic: str
    title: str
    body: str
    data: Optional[Dict[str, str]] = None

@router.post("/send")
async def send_notification(
    request: SendNotificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send notification to a single user"""
    try:
        # Get user's FCM token from database
        query = text("SELECT fcm_token FROM users WHERE id = :user_id")
        result = await db.execute(query, {"user_id": request.user_id})
        user = result.fetchone()
        
        if not user or not user.fcm_token:
            raise HTTPException(status_code=404, detail="User FCM token not found")
        
        # Send notification
        success = await fcm_service.send_notification(
            token=user.fcm_token,
            title=request.title,
            body=request.body,
            data=request.data
        )
        
        if success:
            return {"status": "success", "message": "Notification sent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-multicast")
async def send_multicast_notification(
    request: SendMulticastRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send notification to multiple users"""
    try:
        # Get FCM tokens from database
        query = text("""
            SELECT fcm_token 
            FROM users 
            WHERE id = ANY(:user_ids) AND fcm_token IS NOT NULL
        """)
        result = await db.execute(query, {"user_ids": request.user_ids})
        tokens = [row.fcm_token for row in result.fetchall()]
        
        if not tokens:
            raise HTTPException(status_code=404, detail="No valid FCM tokens found")
        
        # Send multicast notification
        response = await fcm_service.send_multicast_notification(
            tokens=tokens,
            title=request.title,
            body=request.body,
            data=request.data
        )
        
        return {
            "status": "success",
            "success_count": response["success_count"],
            "failure_count": response["failure_count"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-topic")
async def send_topic_notification(request: SendTopicRequest):
    """Send notification to a topic"""
    try:
        success = await fcm_service.send_topic_notification(
            topic=request.topic,
            title=request.title,
            body=request.body,
            data=request.data
        )
        
        if success:
            return {"status": "success", "message": "Topic notification sent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_notification(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Test notification endpoint"""
    try:
        # Get user's FCM token
        query = text("SELECT fcm_token FROM users WHERE id = :user_id")
        result = await db.execute(query, {"user_id": user_id})
        user = result.fetchone()
        
        if not user or not user.fcm_token:
            raise HTTPException(status_code=404, detail="User FCM token not found")
        
        # Send test notification
        success = await fcm_service.send_notification(
            token=user.fcm_token,
            title="🧪 Test Notification",
            body="This is a test notification from MedSight!",
            data={"type": "test"}
        )
        
        if success:
            return {"status": "success", "message": "Test notification sent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test notification")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
