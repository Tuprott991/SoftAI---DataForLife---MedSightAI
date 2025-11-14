from generate_report import generate_clinical_report_from_path

patient_metadata = {
    "patient_id": "P0001",
    "age": 34,
    "sex": "F",
    "study_id": "S-P0001-2025-11-01",
    "image_filename": "h0001.png",
    "image_type": "PA",
    "views": "PA",
    "image_height": 2048,
    "image_width": 2048,
    "source": "test",
    "bbox": "none",
    "target": "no",
    "disease_type": "healthy",
    "indication": "Evaluation of chest symptoms.",
    "comparison_info": "None",
}

image_path = r"Images\h0001.png"   # đổi path tùy máy bạn

result = generate_clinical_report_from_path(image_path, patient_metadata)

print("===== REPORT RESULT =====")
for k, v in result.items():
    print(k, ":", v)
