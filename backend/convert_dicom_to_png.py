"""
Convert all DICOM images to PNG format
Downloads DICOM files from S3, converts to PNG, uploads back, and updates database
"""
import os
import sys
import io
from pathlib import Path

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.config.database import SessionLocal
from app.models.models import Case
from app.services.s3_service import s3_service
import pydicom
from PIL import Image
import numpy as np


def dicom_to_png_bytes(dicom_bytes: bytes) -> bytes:
    """
    Convert DICOM bytes to PNG bytes
    
    Args:
        dicom_bytes: Raw DICOM file bytes
    
    Returns:
        PNG image bytes
    """
    # Load DICOM from bytes
    dicom_file = pydicom.dcmread(io.BytesIO(dicom_bytes))
    
    # Get pixel array
    pixel_array = dicom_file.pixel_array
    
    # Normalize to 0-255 range
    pixel_array = pixel_array.astype(float)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
    pixel_array = pixel_array.astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(pixel_array)
    
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return buffer.read()


def extract_s3_key_from_url(url: str) -> str:
    """
    Extract S3 key from full URL
    
    Args:
        url: Full S3 URL (https://bucket.s3.region.amazonaws.com/key)
    
    Returns:
        S3 key (cases/patient_id/filename.dicom)
    """
    if url.startswith('http'):
        # URL format: https://bucket.s3.region.amazonaws.com/key
        parts = url.split('/')
        # Skip https:, empty, bucket.s3.region.amazonaws.com
        key = '/'.join(parts[3:])
        return key
    return url


def main():
    """Main function to convert all DICOM images to PNG"""
    
    print("="*80)
    print("🔄 DICOM to PNG Converter")
    print("="*80)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Get all cases
        cases = db.query(Case).all()
        total_cases = len(cases)
        
        print(f"\n📊 Found {total_cases} cases in database")
        
        if total_cases == 0:
            print("❌ No cases found")
            return
        
        # Confirm before proceeding
        response = input(f"\n⚠️  This will download {total_cases} DICOM files, convert to PNG, upload, and update database.\nContinue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled")
            return
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for idx, case in enumerate(cases, 1):
            try:
                print(f"\n[{idx}/{total_cases}] Processing case {case.id}...")
                
                # Check if already PNG
                if case.image_path and case.image_path.endswith('.png'):
                    print(f"  ⏭️  Already PNG, skipping...")
                    skipped_count += 1
                    continue
                
                # Extract S3 key from URL
                dicom_s3_key = extract_s3_key_from_url(case.image_path)
                print(f"  📥 Downloading DICOM: {dicom_s3_key}")
                
                # Download DICOM from S3
                dicom_bytes = s3_service.download_file(dicom_s3_key)
                
                # Convert DICOM to PNG
                print(f"  🔄 Converting DICOM to PNG...")
                png_bytes = dicom_to_png_bytes(dicom_bytes)
                
                # Generate new S3 key for PNG
                # Format: cases/{patient_id}.png
                png_filename = f"{case.patient_id}.png"
                png_s3_key = f"cases/{png_filename}"
                
                # Upload PNG to S3
                print(f"  📤 Uploading PNG: {png_s3_key}")
                png_url = s3_service.upload_bytes(
                    file_bytes=png_bytes,
                    filename=png_filename,
                    prefix="cases/",
                    content_type="image/png"
                )
                
                # Update database
                print(f"  💾 Updating database...")
                case.image_path = png_url
                db.commit()
                
                print(f"  ✅ Success! New URL: {png_url[:80]}...")
                
                # Optional: Delete old DICOM file from S3
                # Uncomment if you want to clean up DICOM files
                # try:
                #     s3_service.delete_file(dicom_s3_key)
                #     print(f"  🗑️  Deleted old DICOM file")
                # except:
                #     print(f"  ⚠️  Could not delete old DICOM file")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                db.rollback()
                failed_count += 1
                continue
        
        print("\n" + "="*80)
        print(f"✅ Successfully converted: {success_count} cases")
        print(f"⏭️  Skipped (already PNG): {skipped_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📊 Total: {total_cases}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        db.rollback()
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
