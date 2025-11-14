from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json

from cxr_report_pipeline import generate_clinical_report_from_bytes

app = FastAPI()


@app.post("/generate_report")
async def generate_report(
    file: UploadFile = File(...),
    metadata: str = Form(...),  # JSON string
):
    image_bytes = await file.read()
    patient_metadata = json.loads(metadata)
    report_dict = generate_clinical_report_from_bytes(image_bytes, patient_metadata)
    return JSONResponse(report_dict)
