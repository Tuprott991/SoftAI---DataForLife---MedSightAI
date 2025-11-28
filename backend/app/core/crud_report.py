"""
CRUD operations for Report model
"""
from typing import Optional
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.crud import CRUDBase
from app.models.models import Report


class CRUDReport(CRUDBase[Report]):
    """CRUD operations for Report"""
    
    def get_by_case(self, db: Session, *, case_id: UUID) -> Optional[Report]:
        """Get report for a case"""
        return (
            db.query(Report)
            .filter(Report.case_id == case_id)
            .order_by(Report.created_at.desc())
            .first()
        )
    
    def update_doctor_report(
        self,
        db: Session,
        *,
        report_id: UUID,
        doctor_report: str
    ) -> Report:
        """Update doctor's report section"""
        report = self.get(db, report_id)
        if report:
            report.doctor_report = doctor_report
            db.commit()
            db.refresh(report)
        return report
    
    def add_feedback(
        self,
        db: Session,
        *,
        report_id: UUID,
        feedback_note: str
    ) -> Report:
        """Add feedback note to report"""
        report = self.get(db, report_id)
        if report:
            report.feedback_note = feedback_note
            db.commit()
            db.refresh(report)
        return report


# Create instance
report = CRUDReport(Report)
