"""
CRUD operations for Patient model
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from app.core.crud import CRUDBase
from app.models.models import Patient
from app.schemas.patient import PatientCreate, PatientUpdate


class CRUDPatient(CRUDBase[Patient]):
    """CRUD operations for Patient"""
    
    def get_by_name(self, db: Session, *, name: str) -> Optional[Patient]:
        """Get patient by name"""
        return db.query(Patient).filter(Patient.name == name).first()
    
    def search_patients(
        self, 
        db: Session, 
        *, 
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Patient]:
        """Search patients by name"""
        return (
            db.query(Patient)
            .filter(Patient.name.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
            .all()
        )


# Create instance
patient = CRUDPatient(Patient)
