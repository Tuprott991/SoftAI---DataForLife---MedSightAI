"""
Similarity Search API endpoints
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.core import case as crud_case, ai_result as crud_ai_result
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
    
    Supports three search modes:
    1. case_id: Find cases similar to an existing case
    2. image_path: Find cases similar to an uploaded image
    3. text_description: Find cases matching a text description
    
    Args:
        request: Search request with one of the three search modes
        db: Database session
    
    Returns:
        List of similar cases with similarity scores
    
    Raises:
        400: No search criteria provided
        404: Case not found
        500: Failed to search or retrieve embeddings
    """
    similar_case_ids = []
    similarity_scores = []
    
    if request.case_id:
        # Search by existing case
        case = crud_case.get(db, UUID(request.case_id))
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get embedding from Zilliz or generate new one
        existing_embedding = zilliz_service.get_by_case_id(request.case_id)
        
        if existing_embedding and existing_embedding.get('img_emb'):
            # Use existing image embedding
            image_embedding = existing_embedding['img_emb']
        else:
            # Generate new embedding
            image_path = case.processed_img_path or case.image_path
            
            # Extract S3 key from full URL if necessary
            if image_path.startswith('http'):
                s3_key = '/'.join(image_path.split('/')[3:])
            else:
                s3_key = image_path
            
            image_bytes = s3_service.download_file(s3_key)
            image_embedding = medsigclip_service.generate_image_embedding(image_bytes)
        
        # Search for similar cases by image
        primary_keys, similarity_scores = zilliz_service.search_similar_by_image(
            image_embedding, top_k=request.top_k
        )
        
        # Convert primary keys back to case IDs
        similar_case_ids = _convert_primary_keys_to_case_ids(primary_keys, db)
    
    elif request.image_path:
        # Search by uploaded image
        # Extract S3 key from full URL if necessary
        if request.image_path.startswith('http'):
            s3_key = '/'.join(request.image_path.split('/')[3:])
        else:
            s3_key = request.image_path
        
        image_bytes = s3_service.download_file(s3_key)
        image_embedding = medsigclip_service.generate_image_embedding(image_bytes)
        
        # Search for similar cases by image
        primary_keys, similarity_scores = zilliz_service.search_similar_by_image(
            image_embedding, top_k=request.top_k
        )
        
        # Convert primary keys back to case IDs
        similar_case_ids = _convert_primary_keys_to_case_ids(primary_keys, db)
    
    elif request.text_description:
        # Search by text description
        text_embedding = medsigclip_service.generate_text_embedding(request.text_description)
        
        # Search for similar cases by text
        primary_keys, similarity_scores = zilliz_service.search_similar_by_text(
            text_embedding, top_k=request.top_k
        )
        
        # Convert primary keys back to case IDs
        similar_case_ids = _convert_primary_keys_to_case_ids(primary_keys, db)
    
    else:
        raise HTTPException(
            status_code=400, 
            detail="Must provide case_id, image_path, or text_description"
        )
    
    # Get case details for similar cases
    case_details = []
    for case_id_str in similar_case_ids:
        try:
            case = crud_case.get(db, UUID(case_id_str))
            if case:
                case_details.append(case)
        except:
            continue
    
    return SimilaritySearchResponse(
        similar_cases=case_details,
        similarity_scores=similarity_scores
    )


def _convert_primary_keys_to_case_ids(primary_keys: list, db: Session) -> list:
    """
    Helper function to convert Zilliz primary keys back to case IDs
    This requires querying the database to find cases
    
    Note: This is a workaround. Ideally, Zilliz should store the case_id as a scalar field
    """
    import hashlib
    
    case_ids = []
    
    # Get all cases from database
    all_cases = db.query(crud_case.model).all()
    
    # Create mapping of primary_key -> case_id
    pk_to_case_id = {}
    for case in all_cases:
        case_id_str = str(case.id)
        # Use same hash function as ZillizService._uuid_to_int
        hash_val = int(hashlib.sha256(case_id_str.encode()).hexdigest(), 16)
        primary_key = hash_val % (2**63 - 1)
        pk_to_case_id[primary_key] = case_id_str
    
    # Convert primary keys to case IDs
    for pk in primary_keys:
        if pk in pk_to_case_id:
            case_ids.append(pk_to_case_id[pk])
    
    return case_ids


@router.post("/embed")
async def generate_embeddings(
    case_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Generate and store embeddings for a case
    
    Downloads image from S3, generates MedSigLip embeddings for both image and text,
    and stores them in Zilliz vector database for similarity search.
    
    Args:
        case_id: UUID of the case to generate embeddings for
        db: Database session
    
    Returns:
        Success message with embedding dimensions
    
    Raises:
        404: Case not found
        500: Failed to download image, generate embeddings, or store in Zilliz
    """
    # Get case from database
    case = crud_case.get(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    try:
        # Download image from S3
        # Use processed image if available, otherwise use original
        image_path = case.processed_img_path or case.image_path
        
        # Extract S3 key from full URL if necessary
        if image_path.startswith('http'):
            # URL format: https://bucket.s3.region.amazonaws.com/key
            # Extract key part after bucket domain
            s3_key = '/'.join(image_path.split('/')[3:])
        else:
            s3_key = image_path
        
        image_bytes = s3_service.download_file(s3_key)
        
        # Get diagnosis/description for text embedding
        # Combine case diagnosis, findings, and AI predicted diagnosis
        text_parts = []
        
        if case.diagnosis:
            text_parts.append(f"Diagnosis: {case.diagnosis}")
        
        if case.findings:
            text_parts.append(f"Findings: {case.findings}")
        
        # Get AI result for additional context
        ai_result = crud_ai_result.get_by_case(db, case_id=case_id)
        if ai_result and ai_result.predicted_diagnosis:
            text_parts.append(f"AI Prediction: {ai_result.predicted_diagnosis}")
        
        # Create text description (default to empty if no info available)
        text_description = " | ".join(text_parts) if text_parts else "chest x-ray image"
        
        # Generate embeddings using MedSigLip
        image_emb, text_emb = medsigclip_service.generate_embeddings(
            image_bytes=image_bytes,
            text=text_description
        )
        
        # Store embeddings in Zilliz
        success = zilliz_service.upsert_embedding(
            case_id=str(case_id),
            txt_embedding=text_emb,
            img_embedding=image_emb
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store embeddings in Zilliz"
            )
        
        return {
            "status": "success",
            "message": "Embeddings generated and stored successfully",
            "case_id": str(case_id),
            "image_embedding_dim": len(image_emb),
            "text_embedding_dim": len(text_emb),
            "text_used": text_description
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )


@router.post("/embed/batch")
async def batch_generate_embeddings(
    db: Session = Depends(get_db),
    skip_existing: bool = True
):
    """
    Batch generate embeddings for ALL cases in the database
    
    This endpoint processes all cases in the database and generates embeddings
    for each case. Useful for initial setup or bulk re-indexing.
    
    Args:
        db: Database session
        skip_existing: If True, skip cases that already have embeddings in Zilliz
    
    Returns:
        Summary of batch processing results
        
    Note:
        This can be a long-running operation for large datasets.
        Consider using async task queues (Celery) for production use.
    """
    # Get all cases from database
    all_cases = db.query(crud_case.model).all()
    
    total_cases = len(all_cases)
    processed = 0
    skipped = 0
    failed = 0
    errors = []
    
    print(f"📊 Starting batch embedding generation for {total_cases} cases...")
    
    for case in all_cases:
        case_id_str = str(case.id)
        
        try:
            # Check if embeddings already exist
            if skip_existing:
                existing = zilliz_service.get_by_case_id(case_id_str)
                if existing and existing.get('img_emb'):
                    print(f"⏭️  Skipping case {case_id_str} (already has embeddings)")
                    skipped += 1
                    continue
            
            # Download image from S3
            image_path = case.processed_img_path or case.image_path
            
            if image_path.startswith('http'):
                s3_key = '/'.join(image_path.split('/')[3:])
            else:
                s3_key = image_path
            
            image_bytes = s3_service.download_file(s3_key)
            
            # Build text description
            text_parts = []
            
            if case.diagnosis:
                text_parts.append(f"Diagnosis: {case.diagnosis}")
            
            if case.findings:
                text_parts.append(f"Findings: {case.findings}")
            
            # Get AI result
            ai_result = crud_ai_result.get_by_case(db, case_id=case.id)
            if ai_result and ai_result.predicted_diagnosis:
                text_parts.append(f"AI Prediction: {ai_result.predicted_diagnosis}")
            
            text_description = " | ".join(text_parts) if text_parts else "chest x-ray image"
            
            # Generate embeddings
            image_emb, text_emb = medsigclip_service.generate_embeddings(
                image_bytes=image_bytes,
                text=text_description
            )
            
            # Store in Zilliz
            success = zilliz_service.upsert_embedding(
                case_id=case_id_str,
                txt_embedding=text_emb,
                img_embedding=image_emb
            )
            
            if success:
                processed += 1
                print(f"✅ Processed case {case_id_str} ({processed}/{total_cases})")
            else:
                failed += 1
                error_msg = f"Failed to store embeddings for case {case_id_str}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
                
        except Exception as e:
            failed += 1
            error_msg = f"Case {case_id_str}: {str(e)}"
            errors.append(error_msg)
            print(f"❌ Error processing case {case_id_str}: {str(e)}")
            continue
    
    return {
        "status": "completed",
        "summary": {
            "total_cases": total_cases,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "success_rate": f"{(processed / total_cases * 100):.2f}%" if total_cases > 0 else "0%"
        },
        "errors": errors[:10] if errors else []  # Return first 10 errors
    }
