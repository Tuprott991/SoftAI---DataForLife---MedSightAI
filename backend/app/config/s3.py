"""
AWS S3 configuration
"""
import boto3
from botocore.client import Config
from app.config.settings import settings


def get_s3_client():
    """
    Create and return S3 client
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
        config=Config(signature_version='s3v4')
    )
    return s3_client


def get_s3_resource():
    """
    Create and return S3 resource
    """
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    return s3_resource
