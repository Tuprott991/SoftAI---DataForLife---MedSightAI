"""
Report API endpoints
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import report as crud_report, case as crud_case, patient as crud_patient, ai_result as crud_ai_result
from app.schemas import (
    ReportCreate, ReportUpdate, ReportResponse,
    ReportGenerationRequest, MessageResponse
)
from app.services import medgemma_service

router = APIRouter()


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate medical report using MedGemma LLM
    
    TODO: Call medgemma_service.generate_medical_report()
    """
    case = crud_case.get(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get AI results
    ai_result = crud_ai_result.get_by_case(db, case_id=request.case_id)
    if not ai_result:
        raise HTTPException(status_code=404, detail="AI analysis not found. Run analysis first")
    
    # Get patient info
    patient = crud_patient.get(db, case.patient_id)
    
    # Prepare data for report generation
    image_findings = {
        "predicted_diagnosis": ai_result.predicted_diagnosis,
        "confidence_score": ai_result.confident_score,
        "bounding_boxes": ai_result.bounding_box,
    }
    
    patient_history = {
        "age": patient.age,
        "gender": patient.gender,
        "history": patient.history
    } if patient else None
    
    # TODO: Generate report using MedGemma
    # model_report = medgemma_service.generate_medical_report(
    #     image_findings=image_findings,
    #     patient_history=patient_history,
    #     clinical_context=None
    # )
    
    # Save report
    # report_data = {
    #     "case_id": request.case_id,
    #     "model_report": model_report
    # }
    # report = crud_report.create(db, obj_in=report_data)
    
    raise NotImplementedError("Connect to MedGemma service")


@router.get("/{case_id}", response_model=ReportResponse)
def get_report(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """Get report for a case"""
    report = crud_report.get_by_case(db, case_id=case_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.put("/{report_id}/doctor-report", response_model=ReportResponse)
def update_doctor_report(
    report_id: UUID,
    doctor_report: str,
    db: Session = Depends(get_db)
):
    """Update doctor's report section (Human-in-the-loop)"""
    report = crud_report.update_doctor_report(
        db,
        report_id=report_id,
        doctor_report=doctor_report
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.put("/{report_id}/feedback", response_model=ReportResponse)
def add_feedback(
    report_id: UUID,
    feedback_note: str,
    db: Session = Depends(get_db)
):
    """Add feedback note for model improvement"""
    report = crud_report.add_feedback(
        db,
        report_id=report_id,
        feedback_note=feedback_note
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.delete("/{report_id}", response_model=MessageResponse)
def delete_report(
    report_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a report"""
    report = crud_report.get(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    crud_report.delete(db, id=report_id)
    return {"message": "Report deleted successfully"}
