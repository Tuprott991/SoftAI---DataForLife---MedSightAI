"""
CRUD operations for Case model
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.crud import CRUDBase
from app.models.models import Case


class CRUDCase(CRUDBase[Case]):
    """CRUD operations for Case"""
    
    def get_by_patient(
        self, 
        db: Session, 
        *, 
        patient_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Case]:
        """Get all cases for a patient"""
        return (
            db.query(Case)
            .filter(Case.patient_id == patient_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_similar_cases(
        self,
        db: Session,
        *,
        case_id: UUID,
        similar_cases: List[str],
        similarity_scores: List[float]
    ) -> Case:
        """Update similar cases for a case"""
        case = self.get(db, case_id)
        if case:
            case.similar_cases = similar_cases
            case.similarity_scores = similarity_scores
            db.commit()
            db.refresh(case)
        return case


# Create instance
case = CRUDCase(Case)
