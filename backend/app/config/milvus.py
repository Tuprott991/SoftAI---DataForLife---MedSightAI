"""
Milvus vector database configuration
"""
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config.settings import settings


def connect_milvus():
    """
    Connect to Milvus server
    """
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT
    )


def disconnect_milvus():
    """
    Disconnect from Milvus server
    """
    connections.disconnect("default")


def create_collection():
    """
    Create Milvus collection for case embeddings if not exists
    Collection schema:
    - case_id: UUID (primary key)
    - image_embedding: Vector (float, dimension based on MedSigLip)
    - text_embedding: Vector (float, dimension based on MedSigLip)
    """
    if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
        print(f"Collection {settings.MILVUS_COLLECTION_NAME} already exists")
        return
    
    fields = [
        FieldSchema(name="case_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_DIMENSION),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_DIMENSION),
    ]
    
    schema = CollectionSchema(fields, description="MedSight AI case embeddings")
    collection = Collection(settings.MILVUS_COLLECTION_NAME, schema)
    
    # Create index for vector search
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("image_embedding", index_params)
    collection.create_index("text_embedding", index_params)
    
    print(f"Collection {settings.MILVUS_COLLECTION_NAME} created successfully")


def get_collection():
    """
    Get Milvus collection instance
    """
    return Collection(settings.MILVUS_COLLECTION_NAME)
