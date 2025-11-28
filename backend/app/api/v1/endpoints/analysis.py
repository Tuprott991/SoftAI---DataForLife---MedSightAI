"""
AI Analysis API endpoints
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import case as crud_case, ai_result as crud_ai_result
from app.schemas import (
    AIAnalysisRequest, AIAnalysisResponse, AIResultResponse,
    AIResultCreate, MessageResponse
)
from app.services import (
    ai_model_service, medsigclip_service, 
    zilliz_service, s3_service
)

router = APIRouter()


@router.post("/full-pipeline", response_model=AIAnalysisResponse)
async def run_full_analysis(
    request: AIAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Run complete AI analysis pipeline:
    1. Preprocess image
    2. Run inference
    3. Generate heatmap (Grad-CAM)
    4. Extract concepts
    5. Find similar cases
    6. Store results
    
    TODO: Implement by calling functions from ai_model_service
    """
    case = crud_case.get(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: Implement full pipeline
    # Step 1: Preprocess
    # processed_path = ai_model_service.preprocess_image(case.image_path)
    # crud_case.update(db, db_obj=case, obj_in={"processed_img_path": processed_path})
    
    # Step 2: Run inference
    # results = ai_model_service.run_inference(processed_path)
    
    # Step 3: Store AI results
    # ai_result_data = {
    #     "case_id": request.case_id,
    #     "predicted_diagnosis": results['predicted_diagnosis'],
    #     "confident_score": results['confidence_score'],
    #     "bounding_box": results['bounding_boxes']
    # }
    # ai_result = crud_ai_result.create(db, obj_in=ai_result_data)
    
    # Step 4: Generate heatmap if requested
    # heatmap_path = None
    # if request.include_heatmap:
    #     heatmap_path = ai_model_service.generate_gradcam(processed_path)
    
    # Step 5: Extract concepts if requested
    # concepts = None
    # if request.include_concepts:
    #     concepts = ai_model_service.extract_concepts(processed_path, results['bounding_boxes'])
    
    # Step 6: Generate embeddings and find similar cases (in background)
    # background_tasks.add_task(process_similarity_search, request.case_id, processed_path, db)
    
    raise NotImplementedError("Connect to AI model services")


@router.post("/preprocess")
async def preprocess_image(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Preprocess image for AI analysis
    
    TODO: Call ai_model_service.preprocess_image()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Download from S3
    image_bytes = s3_service.download_file(case.image_path)
    
    # TODO: Preprocess
    # processed_path = ai_model_service.preprocess_image(image_bytes)
    
    # Update case with processed image path
    # crud_case.update(db, db_obj=case, obj_in={"processed_img_path": processed_path})
    
    raise NotImplementedError("Connect to preprocessing module")


@router.post("/inference")
async def run_inference(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Run AI model inference
    
    TODO: Call ai_model_service.run_inference()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    if not case.processed_img_path:
        raise HTTPException(status_code=400, detail="Image not preprocessed yet")
    
    # TODO: Run inference
    # results = ai_model_service.run_inference(case.processed_img_path)
    
    # Store results
    # ai_result_data = {
    #     "case_id": case_id,
    #     "predicted_diagnosis": results['predicted_diagnosis'],
    #     "confident_score": results['confidence_score'],
    #     "bounding_box": results['bounding_boxes']
    # }
    # ai_result = crud_ai_result.create(db, obj_in=ai_result_data)
    
    raise NotImplementedError("Connect to inference module")


@router.get("/{case_id}/heatmap")
async def get_heatmap(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Generate and return Grad-CAM heatmap
    
    TODO: Call ai_model_service.generate_gradcam()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: Generate heatmap
    # heatmap_path = ai_model_service.generate_gradcam(case.processed_img_path)
    # presigned_url = s3_service.get_presigned_url(heatmap_path)
    
    raise NotImplementedError("Connect to Grad-CAM module")


@router.get("/{case_id}/concepts")
async def get_concepts(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get concept-based analysis
    
    TODO: Call ai_model_service.extract_concepts()
    """
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
    if not ai_result:
        raise HTTPException(status_code=404, detail="AI analysis not found")
    
    # TODO: Extract concepts
    # concepts = ai_model_service.extract_concepts(
    #     case.processed_img_path,
    #     ai_result.bounding_box
    # )
    
    raise NotImplementedError("Connect to concept extraction module")


def process_similarity_search(case_id: UUID, image_path: str, db: Session):
    """
    Background task to process similarity search
    """
    # TODO: Generate embeddings
    # image_emb, text_emb = medsigclip_service.generate_embeddings(image_path, description)
    
    # Store in Milvus
    # milvus_service.insert_embedding(str(case_id), image_emb, text_emb)
    
    # Search for similar cases
    # similar_ids, scores = milvus_service.search_similar_by_image(image_emb, top_k=5)
    
    # Update case with similar cases
    # crud_case.update_similar_cases(db, case_id=case_id, similar_cases=similar_ids, similarity_scores=scores)
    
    pass
