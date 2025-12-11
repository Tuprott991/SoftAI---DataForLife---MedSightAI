import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CSRModel(nn.Module):
    def __init__(self, num_classes=7, num_prototypes=5, model_name="densenet121", pretrained=True):
        """
        CSR Model với DenseNet121 backbone
        
        Args:
            num_classes: Số class (7 sau khi loại bỏ)
            num_prototypes: Số prototypes mỗi class
            model_name: 'densenet121', 'densenet169', 'densenet201'
            pretrained: Sử dụng ImageNet pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        
        # --- PHẦN 1: BACKBONE (DenseNet121) ---
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(4,)  # Lấy output của block cuối
        )
        feature_info = self.backbone.feature_info.get_dicts()[-1]
        self.feature_dim = feature_info["num_chs"]
        print(f"🔧 Backbone: {model_name}, feature_dim={self.feature_dim}")

        # --- PHẦN 2: CONCEPT HEAD ---
        # C: Concept Head (Tạo CAMs) - output = num_classes
        self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)

        # --- PHẦN 3: PROTOTYPES ---
        self.embedding_dim = 128
        
        # P: Projector (Chiếu feature về không gian contrastive)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_dim)
        )
        
        # Learnable Prototypes: [Num_Classes, Num_Prototypes_Per_Class, Emb_Dim]
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_prototypes, self.embedding_dim)
        )
        
        # --- PHẦN 4: TASK HEAD ---
        # H: Task Head (Dự đoán bệnh từ điểm tương đồng)
        # Input: [Num_Classes * Num_Prototypes]
        self.task_head = nn.Linear(num_classes * num_prototypes, num_classes)

    def get_features_and_cam(self, x):
        """
        Trích xuất features và CAM (Dùng cho Phase 1)
        
        Returns:
            features: [B, C, H, W] - Feature map từ backbone
            attn_logits: [B, K, H, W] - CAM logits cho mỗi class
        """
        # Chuyển grayscale sang RGB nếu cần
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        features = self.backbone(x)[0]          # [B, feature_dim, H, W]
        attn_logits = self.concept_head(features)  # [B, num_classes, H, W]
        return features, attn_logits

    def get_projected_vectors(self, features, attn_logits):
        """
        Lấy Local Concept Vectors v^k (Dùng cho Phase 2)
        
        Args:
            features: [B, C, H, W]
            attn_logits: [B, K, H, W]
            
        Returns:
            projected_vectors: [B, K, Emb_Dim] - Normalized projected vectors
        """
        B, C, H, W = features.shape
        K = attn_logits.shape[1]  # num_classes
        
        # 1. Normalize CAM (Spatial Softmax)
        attn_weights = F.softmax(attn_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
        
        # 2. Weighted Sum để lấy vector đại diện cho từng concept
        # features: [B, C, H*W] -> [B, H*W, C]
        features_flat = features.view(B, C, -1).permute(0, 2, 1)
        
        # v = weights * features -> [B, K, C]
        local_concept_vectors = torch.bmm(attn_weights.view(B, K, -1), features_flat)
        
        # 3. Project sang không gian embedding -> v'
        projected_vectors = self.projector(local_concept_vectors)  # [B, K, Emb_Dim]
        
        return F.normalize(projected_vectors, p=2, dim=-1)

    def forward(self, x):
        """
        Full forward pass (Dùng cho Phase 3 & Inference)
        
        Returns:
            dict với:
            - logits: [B, num_classes] - Final predictions
            - attn_maps: [B, K, H, W] - CAM maps
            - projected_vectors: [B, K, Emb_Dim]
            - sim_scores: [B, K, M] - Similarity với prototypes
        """
        # 1. Trích xuất features & CAM
        features, attn_logits = self.get_features_and_cam(x)
        
        # 2. Get projected vectors
        projected_vectors = self.get_projected_vectors(features, attn_logits)  # [B, K, Emb_Dim]
        
        # 3. Tính Similarity Score với Prototypes
        # Prototypes: [K, M, Emb] -> normalize
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # Cosine Similarity: [B, K, M]
        sim_scores = torch.einsum('bkc,kmc->bkm', projected_vectors, prototypes_norm)
        
        # 4. Flatten thành vector s [B, K*M]
        s_vector = sim_scores.reshape(x.size(0), -1)
        
        # 5. Predict từ similarity scores
        logits = self.task_head(s_vector)
        
        return {
            "logits": logits,
            "attn_maps": attn_logits,
            "projected_vectors": projected_vectors,
            "sim_scores": sim_scores
        }


# # Test model
# if __name__ == "__main__":
#     # Test với input giả
#     model = CSRModel(num_classes=7, num_prototypes=15, model_name="densenet121")
    
#     # Input: [Batch, Channel, H, W]
#     x = torch.randn(2, 1, 384, 384)  # Grayscale input
    
#     output = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Logits shape: {output['logits'].shape}")           # [2, 7]
#     print(f"Attn maps shape: {output['attn_maps'].shape}")     # [2, 7, 12, 12]
#     print(f"Projected vectors: {output['projected_vectors'].shape}")  # [2, 7, 128]
#     print(f"Sim scores: {output['sim_scores'].shape}")         # [2, 7, 15]