"""
CRUD operations for AIResult model
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.crud import CRUDBase
from app.models.models import AIResult


class CRUDAIResult(CRUDBase[AIResult]):
    """CRUD operations for AIResult"""
    
    def get_by_case(self, db: Session, *, case_id: UUID) -> Optional[AIResult]:
        """Get AI result for a case (most recent)"""
        return (
            db.query(AIResult)
            .filter(AIResult.case_id == case_id)
            .order_by(AIResult.created_at.desc())
            .first()
        )
    
    def get_all_by_case(self, db: Session, *, case_id: UUID) -> List[AIResult]:
        """Get all AI results for a case"""
        return (
            db.query(AIResult)
            .filter(AIResult.case_id == case_id)
            .order_by(AIResult.created_at.desc())
            .all()
        )


# Create instance
ai_result = CRUDAIResult(AIResult)
