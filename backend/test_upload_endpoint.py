"""
Quick test for /api/v1/cases/upload endpoint
Upload X-ray image to S3 and create case
"""
import requests
import os

BASE_URL = "http://localhost:8000/api/v1"

def load_test_image():
    """Load real test image from current directory"""
    image_path = os.path.join(os.path.dirname(__file__), "test.png")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Read image file
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    return image_data

# Step 1: Create a patient first
print("\n1️⃣ Creating test patient...")
patient_data = {
    "name": "Test Patient 2",
    "age": 45,
    "gender": "male"
}

response = requests.post(f"{BASE_URL}/patients/", json=patient_data)

if response.status_code == 201:
    patient = response.json()
    patient_id = patient['id']
    print(f"✅ Patient created: {patient_id}")
else:
    print(f"❌ Failed to create patient: {response.status_code}")
    print(response.text)
    exit(1)

# Step 2: Upload image using /cases/upload endpoint
print(f"\n2️⃣ Uploading X-ray image to /api/v1/cases/upload...")

image_data = load_test_image()

files = {
    'file': ('test.png', image_data, 'image/png')
}

params = {
    'patient_id': patient_id
}

response = requests.post(
    f"{BASE_URL}/cases/upload",
    files=files,
    params=params
)

print(f"\n📊 Response Status: {response.status_code}")
print(f"📄 Response Body:")
print(response.text)

if response.status_code == 201:
    case = response.json()
    print(f"\n✅ SUCCESS! Case created:")
    print(f"   Case ID: {case['id']}")
    print(f"   Patient ID: {case['patient_id']}")
    print(f"   Image URL (Public): {case['image_path']}")
    print(f"   Timestamp: {case['timestamp']}")
    print(f"\n🌐 This URL is permanent and publicly accessible!")
    print(f"   Copy and paste in browser to view: {case['image_path']}")
    
    # Step 3: Verify we can get the case
    print(f"\n3️⃣ Verifying case retrieval...")
    case_id = case['id']
    response = requests.get(f"{BASE_URL}/cases/{case_id}")
    
    if response.status_code == 200:
        print(f"✅ Case retrieved successfully")
        
        # Step 4: Get presigned URL
        print(f"\n4️⃣ Getting presigned URL for image...")
        response = requests.get(f"{BASE_URL}/cases/{case_id}/image-url")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Presigned URL generated:")
            print(f"   URL: {result['url'][:80]}...")
            print(f"   Expires in: {result['expires_in']} seconds")
        else:
            print(f"⚠️ Failed to get presigned URL: {response.status_code}")
    else:
        print(f"⚠️ Failed to retrieve case: {response.status_code}")
else:
    print(f"\n❌ FAILED!")
    print(f"Error details: {response.json() if response.headers.get('content-type') == 'application/json' else response.text}")
