import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # Khuyên dùng timm cho linh hoạt backbone


class MedicalConceptModel(nn.Module):
    def __init__(self, num_classes=14, model_name="resnet50", pretrained=True):
        super().__init__()

        # 1. Backbone: Trích xuất đặc trưng không gian (Spatial Features)
        # Output mong muốn: [Batch, Features, H_feat, W_feat]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),  # Lấy output của layer cuối cùng
        )

        # Lấy thông tin số kênh (channels) của backbone
        feature_info = self.backbone.feature_info.get_dicts()[-1]
        self.feature_dim = feature_info["num_chs"]  # Ví dụ ResNet50 là 2048

        # 2. Concept Head (Attention Mechanism)
        # Tạo ra K bản đồ nhiệt cho K lớp bệnh
        # Input: Feature_Dim -> Output: Num_Classes (1 map per class)
        self.concept_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)

        # 3. Projector & Classifier
        # Project vector đặc trưng về không gian nhỏ hơn để tính Cosine Similarity hiệu quả
        self.embedding_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),  # Output dimension cho Contrastive
        )

        # Classifier: Dự đoán xác suất bệnh từ Concept Vector
        self.classifier = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        # x: [Batch, 1, 384, 384] (Grayscale)
        # Backbone thường cần 3 kênh màu, ta repeat channel 1 lên 3 lần
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. Spatial Features: [B, C, H, W] (Ví dụ: [B, 2048, 12, 12])
        features = self.backbone(x)[0]
        B, C, H, W = features.shape

        # 2. Attention Maps (Logits): [B, Num_Classes, H, W]
        attn_logits = self.concept_head(features)

        # Chuyển thành trọng số (Softmax trên không gian H,W để tìm vùng quan trọng nhất)
        # [B, K, H*W]
        attn_weights = F.softmax(attn_logits.view(B, -1, H * W), dim=-1)
        attn_maps = attn_weights.view(B, -1, H, W)

        # 3. Concept Aggregation (Weighted Average Pooling)
        # Ta nhân đặc trưng với attention map để lấy ra vector đại diện cho từng bệnh
        # Features: [B, C, H*W] -> [B, H*W, C] (transpose)
        features_flat = features.view(B, C, -1).permute(0, 2, 1)

        # Concept Vectors = Attn_Weights x Features
        # [B, K, H*W] x [B, H*W, C] -> [B, K, C]
        concept_vectors_raw = torch.bmm(attn_weights, features_flat)

        B, K, C = concept_vectors_raw.shape
        concept_vectors_flat = concept_vectors_raw.view(B * K, C)

        # 4. Projection & Classification
        # [B, K, C] -> [B, K, Embedding_Dim]
        concept_embeddings = self.projector(concept_vectors_flat)

        # Normalize vectors để dùng cho Cosine Similarity Loss/Inference
        concept_embeddings = concept_embeddings.view(B, K, -1)

        # Normalize L2
        concept_embeddings = F.normalize(concept_embeddings, p=2, dim=-1)

        # Classification Logits: [B, K, 1] -> [B, K]
        logits = self.classifier(concept_embeddings).squeeze(-1)

        return {
            "logits": logits,  # [B, K] - Dùng cho Cls Loss
            "attn_maps": attn_logits,  # [B, K, H, W] - Dùng cho Seg/Bbox Loss (lưu ý trả về logits chưa qua softmax cho loss)
            "concept_vectors": concept_embeddings,  # [B, K, Emb_Dim] - Dùng cho Cosine/Contrastive
        }
