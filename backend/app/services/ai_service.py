"""
AI Model Integration Service
This service interfaces with AI models developed by other team members
"""
import sys
import os
from typing import Dict, Any, List, Tuple
from fastapi import HTTPException
from app.config.settings import settings
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import io



# Add model paths to system path
sys.path.append(os.path.abspath(settings.MODEL_INFERENCE_PATH))
sys.path.append(os.path.abspath(settings.MEDGEMMA_PATH))
sys.path.append(os.path.abspath(settings.VINDR_DATASET_PATH))


class AIModelService:
    """Service for AI model inference and analysis"""
    
    def __init__(self):
        """
        Initialize AI models
        
        TODO: Import and initialize models from other developers' code
        Example:
            from MedSightAI.inference import load_model, predict
            from MedSightAI.model import DenseNet121Model
            self.model = load_model()
        """
        pass
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess X-ray image for model input
        
        Args:
            image_path: Path to original image (S3 or local)
        
        Returns:
            Path to preprocessed image
        
        TODO: Call preprocessing function from MedSightAI/src/dataset.py
        Example:
            from MedSightAI.src.dataset import preprocess_xray
            processed_img = preprocess_xray(image_path)
            return processed_img_path
        """
        raise NotImplementedError("Connect to MedSightAI preprocessing module")
    
    def run_inference(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Run AI model inference on preprocessed image
        
        Args:
            image_path: Path to preprocessed image
        
        Returns:
            Dictionary containing:
                - predicted_diagnosis: str
                - confidence_score: float
                - bounding_boxes: List[Dict] with lesion locations
                - lesion_types: List[str]
        
        TODO: Call inference function from MedSightAI/inference.py
        Example:
            from MedSightAI.inference import run_inference
            results = run_inference(self.model, image_path)
            return {
                'predicted_diagnosis': results['diagnosis'],
                'confidence_score': results['confidence'],
                'bounding_boxes': results['boxes'],
                'lesion_types': results['lesion_classes']
            }
        """
        raise NotImplementedError("Connect to MedSightAI inference module")
    
    def generate_gradcam(
        self,
        image_path: str,
        target_layer: str = "features.denseblock4"
    ) -> str:
        """
        Generate Grad-CAM heatmap for explainability
        
        Args:
            image_path: Path to image
            target_layer: Target layer for Grad-CAM
        
        Returns:
            Path to heatmap image
        
        TODO: Implement Grad-CAM generation
        Example:
            from pytorch_grad_cam import GradCAM
            gradcam = GradCAM(model=self.model, target_layers=[target_layer])
            heatmap = gradcam(input_tensor)
            # Save heatmap and return path
        """
        raise NotImplementedError("Implement Grad-CAM generation")
    
    def extract_concepts(
        self,
        image_path: str,
        bounding_boxes: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Extract pathological concepts from detected regions
        
        Args:
            image_path: Path to image
            bounding_boxes: Detected bounding boxes
        
        Returns:
            List of concepts with confidence scores
            [
                {
                    'concept': 'consolidation',
                    'confidence': 0.85,
                    'region': bbox
                }
            ]
        
        TODO: Implement concept-based prototype analysis
        This should map detected regions to medical concepts like:
        - consolidation, cavity, fibrosis, opacity, nodule, etc.
        """
        raise NotImplementedError("Implement concept extraction")


class MedSigLipService:
    """Service for MedSigLip embedding generation"""
    
    def __init__(self):
        """
        Initialize MedSigLip model
        Loads model from HuggingFace: aysangh/medsiglip-448-vindr-bin
        """
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self._initialized = False
    
    def _lazy_load(self):
        """Lazy load the model when first needed to save memory"""
        if self._initialized:
            return
        
        try:            
            model_name = "aysangh/medsiglip-448-vindr-bin"
            base_model = "google/siglip-base-patch16-224"
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading MedSigLip model on {self.device}...")
            
            # Load tokenizer and image processor from base model first
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.image_processor = AutoImageProcessor.from_pretrained(base_model)
            
            # Load the fine-tuned model
            # This is a custom trained model, load it as a generic AutoModel
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            self._initialized = True
            print("MedSigLip model loaded successfully")
        except Exception as e:
            print(f"Error loading MedSigLip model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load MedSigLip model: {str(e)}"
            )
    
    def generate_image_embedding(self, image_bytes: bytes) -> List[float]:
        """
        Generate image embedding using MedSigLip
        
        Args:
            image_bytes: Image data as bytes
        
        Returns:
            Image embedding vector (1152 dimensions)
        """
        self._lazy_load()
        
        try:            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Process image
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
                size={"height": 448, "width": 448}
            ).pixel_values.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                outputs = self.model.vision_model(pixel_values=pixel_values)
                
                # Get pooled output or first token
                if hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state[:, 0]
                else:
                    embedding = outputs[0][:, 0]
            
            # Convert to list and return
            return embedding.cpu().numpy()[0].tolist()
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate image embedding: {str(e)}"
            )
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding using MedSigLip
        
        Args:
            text: Medical text description
        
        Returns:
            Text embedding vector (1152 dimensions)
        """
        self._lazy_load()
        
        try:
            import torch
            
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                outputs = self.model.text_model(**inputs)
                
                # Get pooled output or first token
                if hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state[:, 0]
                else:
                    embedding = outputs[0][:, 0]
            
            # Convert to list and return
            return embedding.cpu().numpy()[0].tolist()
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate text embedding: {str(e)}"
            )
    
    def generate_embeddings(
        self,
        image_bytes: bytes,
        text: str
    ) -> Tuple[List[float], List[float]]:
        """
        Generate both image and text embeddings
        
        Args:
            image_bytes: Image data as bytes
            text: Medical text description
        
        Returns:
            Tuple of (image_embedding, text_embedding)
        """
        image_emb = self.generate_image_embedding(image_bytes)
        text_emb = self.generate_text_embedding(text)
        return image_emb, text_emb


# Create singleton instances
ai_model_service = AIModelService()
medsigclip_service = MedSigLipService()
