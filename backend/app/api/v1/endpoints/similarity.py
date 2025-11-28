"""
Similarity Search API endpoints
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import case as crud_case
from app.schemas import SimilaritySearchRequest, SimilaritySearchResponse
from app.services import medsigclip_service, zilliz_service, s3_service

router = APIRouter()


@router.post("/search", response_model=SimilaritySearchResponse)
async def search_similar_cases(
    request: SimilaritySearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search for similar cases using MedSigLip embeddings
    Can search by case_id, image_path, or text description
    
    TODO: Implement similarity search workflow
    """
    similar_cases = []
    similarity_scores = []
    
    if request.case_id:
        # Search by existing case
        case = crud_case.get(db, UUID(request.case_id))
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # TODO: Get embedding from Milvus or generate new one
        # existing_embedding = milvus_service.get_embedding(request.case_id)
        # if existing_embedding:
        #     image_embedding = existing_embedding['image_embedding']
        # else:
        #     image_bytes = s3_service.download_file(case.image_path)
        #     image_embedding = medsigclip_service.generate_image_embedding(image_bytes)
        
        # similar_cases, similarity_scores = milvus_service.search_similar_by_image(
        #     image_embedding, top_k=request.top_k
        # )
        
        pass
    
    elif request.image_path:
        # Search by uploaded image
        # TODO: Generate embedding and search
        # image_bytes = s3_service.download_file(request.image_path)
        # image_embedding = medsigclip_service.generate_image_embedding(image_bytes)
        # similar_cases, similarity_scores = milvus_service.search_similar_by_image(
        #     image_embedding, top_k=request.top_k
        # )
        pass
    
    elif request.text_description:
        # Search by text description
        # TODO: Generate text embedding and search
        # text_embedding = medsigclip_service.generate_text_embedding(request.text_description)
        # similar_cases, similarity_scores = milvus_service.search_similar_by_text(
        #     text_embedding, top_k=request.top_k
        # )
        pass
    
    else:
        raise HTTPException(status_code=400, detail="Must provide case_id, image_path, or text_description")
    
    # TODO: Get case details for similar cases
    # case_details = [crud_case.get(db, UUID(case_id)) for case_id in similar_cases]
    
    raise NotImplementedError("Connect to MedSigLip and Milvus services")


@router.post("/embed")
async def generate_embeddings(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Generate and store embeddings for a case
    
    TODO: Call medsigclip_service and milvus_service
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: Download image
    # image_bytes = s3_service.download_file(case.image_path)
    
    # TODO: Generate embeddings
    # Get diagnosis/description for text embedding
    # ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
    # text_description = ai_result.predicted_diagnosis if ai_result else ""
    
    # image_emb, text_emb = medsigclip_service.generate_embeddings(
    #     image_path=image_bytes,
    #     text=text_description
    # )
    
    # TODO: Store in Milvus
    # milvus_service.insert_embedding(str(case_id), image_emb, text_emb)
    
    raise NotImplementedError("Connect to embedding generation")
