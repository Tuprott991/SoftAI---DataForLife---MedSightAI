"""
S3 Storage Service
Handles file upload, download, and deletion from AWS S3
"""
import os
import io
from typing import Optional, BinaryIO
from datetime import datetime
from uuid import uuid4
from fastapi import UploadFile, HTTPException
from app.config.s3 import get_s3_client
from app.config.settings import settings


class S3Service:
    """Service for S3 operations"""
    
    def __init__(self):
        self.client = get_s3_client()
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def upload_file(
        self,
        file: UploadFile,
        prefix: str = "",
        custom_name: Optional[str] = None
    ) -> str:
        """
        Upload a file to S3
        
        Args:
            file: UploadFile object from FastAPI
            prefix: S3 prefix/folder (e.g., 'original/', 'processed/')
            custom_name: Custom filename (optional)
        
        Returns:
            S3 file path
        """
        try:
            # Generate unique filename
            if custom_name:
                filename = custom_name
            else:
                ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid4()}{ext}"
            
            # Construct S3 key
            s3_key = f"{prefix}{filename}"
            
            # Upload to S3
            self.client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
            )
            
            return s3_key
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
    
    def upload_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        prefix: str = "",
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload bytes data to S3
        
        Args:
            file_bytes: Binary data
            filename: Filename
            prefix: S3 prefix/folder
            content_type: Content type
        
        Returns:
            S3 file path
        """
        try:
            s3_key = f"{prefix}{filename}"
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type
            )
            
            return s3_key
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload bytes to S3: {str(e)}")
    
    def download_file(self, s3_key: str) -> bytes:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            File bytes
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found in S3: {str(e)}")
    
    def download_to_stream(self, s3_key: str) -> io.BytesIO:
        """
        Download file to BytesIO stream
        
        Args:
            s3_key: S3 object key
        
        Returns:
            BytesIO stream
        """
        try:
            stream = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, s3_key, stream)
            stream.seek(0)
            return stream
        
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found in S3: {str(e)}")
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file from S3: {str(e)}")
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary file access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
        
        Returns:
            Presigned URL
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def get_file_size(self, s3_key: str) -> int:
        """
        Get file size from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            File size in bytes
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


# Create singleton instance
s3_service = S3Service()
