"""
Zilliz Cloud Configuration
"""
from app.config.settings import settings


def get_zilliz_headers():
    """Get headers for Zilliz Cloud API requests"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.ZILLIZ_CLOUD_API_KEY}"
    }


def get_zilliz_config():
    """Get Zilliz Cloud configuration"""
    return {
        "base_url": settings.ZILLIZ_CLOUD_URI,
        "collection_name": settings.ZILLIZ_COLLECTION_NAME,
        "txt_dimension": settings.ZILLIZ_TXT_DIMENSION,
        "img_dimension": settings.ZILLIZ_IMG_DIMENSION
    }
