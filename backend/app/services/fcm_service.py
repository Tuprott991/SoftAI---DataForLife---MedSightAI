"""
FCM Notification Service
Send push notifications to mobile devices via Firebase Cloud Messaging
"""

import firebase_admin
from firebase_admin import credentials, messaging
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FCMService:
    """Firebase Cloud Messaging Service"""
    
    def __init__(self):
        """Initialize Firebase Admin SDK"""
        # Initialize Firebase Admin (if not already initialized)
        if not firebase_admin._apps:
            try:
                # Use default credentials or service account key
                # Make sure serviceAccountKey.json is in backend/ folder
                cred = credentials.Certificate('serviceAccountKey.json')
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firebase Admin SDK initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing Firebase Admin: {e}")
                raise
    
    async def send_notification(
        self,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send notification to a single device
        
        Args:
            token: FCM device token
            title: Notification title
            body: Notification body
            data: Additional data payload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                token=token,
            )
            
            response = messaging.send(message)
            logger.info(f"✅ Notification sent successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending notification: {e}")
            return False
    
    async def send_multicast_notification(
        self,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None
    ) -> Dict[str, int]:
        """
        Send notification to multiple devices
        
        Args:
            tokens: List of FCM device tokens
            title: Notification title
            body: Notification body
            data: Additional data payload
            
        Returns:
            Dict with success_count and failure_count
        """
        try:
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                tokens=tokens,
            )
            
            response = messaging.send_multicast(message)
            
            logger.info(
                f"✅ Multicast notification sent: "
                f"{response.success_count} success, "
                f"{response.failure_count} failures"
            )
            
            # Log failed tokens
            if response.failure_count > 0:
                failed_tokens = [
                    tokens[idx] for idx, resp in enumerate(response.responses)
                    if not resp.success
                ]
                logger.warning(f"❌ Failed tokens: {failed_tokens}")
            
            return {
                "success_count": response.success_count,
                "failure_count": response.failure_count,
            }
            
        except Exception as e:
            logger.error(f"❌ Error sending multicast notification: {e}")
            return {"success_count": 0, "failure_count": len(tokens)}
    
    async def send_topic_notification(
        self,
        topic: str,
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send notification to a topic
        
        Args:
            topic: FCM topic name
            title: Notification title
            body: Notification body
            data: Additional data payload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                topic=topic,
            )
            
            response = messaging.send(message)
            logger.info(f"✅ Topic notification sent successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending topic notification: {e}")
            return False

# Singleton instance
fcm_service = FCMService()
