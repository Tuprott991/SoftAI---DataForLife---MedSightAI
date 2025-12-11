"""
Create mock data for 300 patients with Vietnamese names and DICOM images
"""
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.config.database import SessionLocal
from app.models.models import Patient, Case
from app.services.s3_service import s3_service
from app.config.s3 import settings
import pydicom
from PIL import Image
import numpy as np
import io

# Vietnamese name pools
LAST_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng",
    "Đỗ", "Ngô", "Hồ", "Dương", "Lý", "Bùi", "Đinh", "Trương", "Mai", "Tạ",
    "Lương", "Quách", "Chu", "Thái"
]

MIDDLE_NAMES = [
    "Văn", "Thị", "Hữu", "Minh", "Gia", "Ngọc", "Quốc", "Phúc", "Thanh", "Xuân",
    "Hoài", "Hải", "Khánh", "Bảo", "Trung", "Anh", "Tấn", "Nhật", "Đức", "Thiên",
    "Thành", "Đình", "Diệu", "Thế"
]

FIRST_NAMES = [
    "An", "Bình", "Dũng", "Huy", "Khang", "Long", "Minh", "Nam", "Phúc", "Quân",
    "Sơn", "Thiện", "Trung", "Tuấn", "Vy", "Lan", "Hà", "Như", "Ngọc", "Hạnh",
    "Ly", "Thảo", "Quỳnh", "Trâm"
]

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

STATUS_OPTIONS = ["stable", "improving", "critical"]

GENDERS = ["Nam", "Nữ"]

# Underlying conditions pool
UNDERLYING_CONDITIONS = [
    {"hypertension": True, "diabetes": False, "asthma": False},
    {"hypertension": False, "diabetes": True, "asthma": False},
    {"hypertension": False, "diabetes": False, "asthma": True},
    {"hypertension": True, "diabetes": True, "asthma": False},
    {"hypertension": True, "diabetes": False, "asthma": True},
    {"hypertension": False, "diabetes": True, "asthma": True},
    {"hypertension": True, "diabetes": True, "asthma": True},
    {"hypertension": False, "diabetes": False, "asthma": False},
]


def generate_vietnamese_name():
    """Generate a random Vietnamese name"""
    last = random.choice(LAST_NAMES)
    middle = random.choice(MIDDLE_NAMES)
    first = random.choice(FIRST_NAMES)
    return f"{last} {middle} {first}"


def generate_phone_number():
    """Generate a random 10-digit Vietnamese phone number starting with 0"""
    # Vietnamese mobile numbers typically start with 09, 08, 07, 03
    prefixes = ['09', '08', '07', '03']
    prefix = random.choice(prefixes)
    # Generate remaining 8 digits
    remaining = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    return f"{prefix}{remaining}"


def generate_random_timestamp(days_back=180):
    """Generate a random timestamp within the last N days"""
    now = datetime.utcnow()
    random_days = random.randint(0, days_back)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    return now - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)


def dicom_to_png_bytes(dicom_bytes: bytes) -> bytes:
    """Convert DICOM bytes to PNG bytes"""
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


def create_mock_patient():
    """Create a patient with random Vietnamese data"""
    patient = Patient(
        name=generate_vietnamese_name(),
        age=random.randint(10, 80),
        gender=random.choice(GENDERS),
        history=None,  # Don't fill anything
        created_at=generate_random_timestamp(days_back=365),  # Random date within last year
        blood_type=random.choice(BLOOD_TYPES),
        status=random.choice(STATUS_OPTIONS),
        underlying_condition=random.choice(UNDERLYING_CONDITIONS),
        phone_number=generate_phone_number(),
        fcm_token=None  # Don't fill anything
    )
    return patient


def upload_dicom_to_s3(dicom_path: Path, patient_id: str) -> str:
    """
    Convert DICOM to PNG and upload to S3
    
    Args:
        dicom_path: Path to DICOM file
        patient_id: Patient UUID for organizing in S3
    
    Returns:
        S3 URL of uploaded PNG file
    """
    # Read DICOM file
    with open(dicom_path, 'rb') as f:
        dicom_bytes = f.read()
    
    # Convert DICOM to PNG
    png_bytes = dicom_to_png_bytes(dicom_bytes)
    
    # Generate PNG filename: {patient_id}.png
    filename = f"{patient_id}.png"
    prefix = "cases/"
    
    # Upload PNG to S3
    s3_url = s3_service.upload_bytes(
        file_bytes=png_bytes,
        filename=filename,
        prefix=prefix,
        content_type="image/png"
    )
    
    return s3_url


def main():
    """Main function to create 300 patients with cases and images"""
    
    # Get all DICOM files from sample_300 folder
    sample_folder = Path(__file__).parent / "sample_300"
    
    if not sample_folder.exists():
        print(f"❌ Error: sample_300 folder not found at {sample_folder}")
        return
    
    dicom_files = sorted(list(sample_folder.glob("*.dicom")))
    
    if len(dicom_files) == 0:
        print(f"❌ Error: No DICOM files found in {sample_folder}")
        return
    
    print(f"📁 Found {len(dicom_files)} DICOM files in sample_300/")
    print(f"📊 Will create {len(dicom_files)} patients with cases\n")
    
    # Confirm before proceeding
    response = input(f"⚠️  This will create {len(dicom_files)} patients and upload {len(dicom_files)} files to S3.\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    # Create database session
    db = SessionLocal()
    
    try:
        success_count = 0
        failed_count = 0
        
        for idx, dicom_file in enumerate(dicom_files, 1):
            try:
                print(f"\n[{idx}/{len(dicom_files)}] Processing {dicom_file.name}...")
                
                # Step 1: Create patient
                patient = create_mock_patient()
                db.add(patient)
                db.commit()
                db.refresh(patient)
                
                print(f"  ✅ Created patient: {patient.name} (ID: {patient.id})")
                print(f"     Age: {patient.age}, Gender: {patient.gender}, Blood: {patient.blood_type}")
                print(f"     Phone: {patient.phone_number}, Status: {patient.status}")
                
                # Step 2: Convert DICOM to PNG and upload to S3
                print(f"  🔄 Converting {dicom_file.name} to PNG...")
                s3_url = upload_dicom_to_s3(dicom_file, str(patient.id))
                print(f"  ✅ Uploaded PNG to: {s3_url[:80]}...")
                
                # Step 3: Create case
                case = Case(
                    patient_id=patient.id,
                    image_path=s3_url,
                    processed_img_path=None,  # Don't fill
                    timestamp=generate_random_timestamp(days_back=90),  # Random within last 3 months
                    similar_cases=None,  # Don't fill
                    similarity_scores=None,  # Don't fill
                    diagnosis=None,  # Don't fill
                    findings=None  # Don't fill
                )
                db.add(case)
                db.commit()
                db.refresh(case)
                
                print(f"  ✅ Created case (ID: {case.id})")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed to process {dicom_file.name}: {str(e)}")
                db.rollback()
                failed_count += 1
                continue
        
        print("\n" + "="*80)
        print(f"✅ Successfully created: {success_count} patients + cases")
        print(f"❌ Failed: {failed_count}")
        print(f"📊 Total: {success_count + failed_count}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        db.rollback()
    
    finally:
        db.close()


if __name__ == "__main__":
    print("="*80)
    print("🏥 MedSight AI - Mock Data Generator")
    print("="*80)
    main()
