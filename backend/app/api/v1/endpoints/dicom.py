"""
DICOM file viewing and conversion endpoint
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
import httpx
import io
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/view")
async def view_dicom(
    url: str = Query(..., description="DICOM file URL from S3")
):
    """
    Convert and serve DICOM file as viewable image
    
    This endpoint:
    1. Downloads DICOM file from S3 URL
    2. Converts it to PNG format
    3. Returns as image response
    """
    try:
        logger.info(f"Fetching DICOM from: {url}")
        
        # Download DICOM file from S3
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to download DICOM: {response.status_code}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Failed to download DICOM file: {response.status_code}"
                )
            
            dicom_data = response.content
            logger.info(f"Downloaded {len(dicom_data)} bytes")
        
        # Convert DICOM to image
        try:
            import pydicom
            from PIL import Image
            import numpy as np
            
            # Read DICOM data
            dicom = pydicom.dcmread(io.BytesIO(dicom_data))
            logger.info("DICOM file parsed successfully")
            
            # Get pixel array
            pixel_array = dicom.pixel_array
            
            # Normalize to 0-255 range
            if pixel_array.dtype != np.uint8:
                pixel_array = pixel_array.astype(np.float64)
                pixel_min = pixel_array.min()
                pixel_max = pixel_array.max()
                if pixel_max > pixel_min:
                    pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
                    pixel_array = (pixel_array * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
            # Apply window/level if available
            if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                try:
                    window_center = float(dicom.WindowCenter) if not isinstance(dicom.WindowCenter, (list, tuple)) else float(dicom.WindowCenter[0])
                    window_width = float(dicom.WindowWidth) if not isinstance(dicom.WindowWidth, (list, tuple)) else float(dicom.WindowWidth[0])
                    
                    img_min = window_center - window_width / 2
                    img_max = window_center + window_width / 2
                    
                    pixel_array = np.clip(pixel_array, img_min, img_max)
                    pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    logger.info(f"Applied window/level: center={window_center}, width={window_width}")
                except Exception as e:
                    logger.warning(f"Failed to apply window/level: {e}")
            
            # Convert to PIL Image
            image = Image.fromarray(pixel_array)
            
            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr.seek(0)
            
            logger.info("Successfully converted DICOM to PNG")
            
            return StreamingResponse(
                img_byte_arr,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    "Content-Disposition": "inline"
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting DICOM: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to convert DICOM file: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing DICOM view request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
