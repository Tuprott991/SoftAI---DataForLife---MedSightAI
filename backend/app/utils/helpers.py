"""
Utility functions
"""
import os
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException
from app.config.settings import settings


def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded image file
    
    Args:
        file: UploadFile from FastAPI
    
    Returns:
        True if valid
    
    Raises:
        HTTPException if invalid
    """
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
        )
    
    return True


def calculate_iou(box1: dict, box2: dict) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1: {"x": x, "y": y, "width": w, "height": h}
        box2: {"x": x, "y": y, "width": w, "height": h}
    
    Returns:
        IoU score (0-1)
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = x1_min + box1["width"]
    y1_max = y1_min + box1["height"]
    
    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = x2_min + box2["width"]
    y2_max = y2_min + box2["height"]
    
    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def format_medical_text(text: str) -> str:
    """Format medical text for display"""
    # Remove extra whitespace
    text = " ".join(text.split())
    return text
