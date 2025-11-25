import torch
import torch.nn as nn
from transformers import AutoModel


class MedicalConceptModel(nn.Module):
    def __init__(self, num_classes=26, model_name="google/siglip-base-patch16-384"):
        super().__init__()

        # 1. Backbone (MedSigLIP)
        # Load pre-trained weights từ HuggingFace
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze backbone ban đầu (tuỳ chọn, mở ra nếu muốn finetune cả backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.hidden_dim = self.backbone.config.vision_config.hidden_size
        self.num_classes = num_classes

        # 2. Concept Head (Tạo ra N bản đồ nhiệt cho N bệnh)
        # Input: 768 -> Output: 26 channels (mỗi channel là 1 heatmap của 1 bệnh)
        self.concept_head = nn.Conv2d(self.hidden_dim, num_classes, kernel_size=1)

        # 3. Projector & Classifier
        # Mỗi concept vector sẽ đi qua projector này
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
        )

        # Classifier cuối cùng: Từ vector 256 chiều -> ra xác suất (logit)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: [Batch, 3, 384, 384]

        # --- BƯỚC 1: BACKBONE FEATURE EXTRACTION ---
        outputs = self.backbone.vision_model(x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # [Batch, Seq_Len, Hidden]

        # Reshape từ Sequence (L) sang Spatial (H, W)
        # Với ảnh 384x384, patch 16 -> grid 24x24
        B, L, C = last_hidden_state.shape
        H = W = int(L**0.5)

        # Features map F: [B, C, H, W]
        features = last_hidden_state.permute(0, 2, 1).reshape(B, C, H, W)

        # --- BƯỚC 2: CONCEPT ATTENTION MAPS ---
        # Tự học vùng quan trọng cho từng bệnh
        # attention_logits: [B, Num_Classes, H, W]
        attention_logits = self.concept_head(features)

        # Chuyển thành xác suất attention (Softmax over spatial dims hoặc Sigmoid)
        # Ở đây dùng Spatial Softmax để model tập trung vào vùng đặc trưng nhất
        B, K, H, W = attention_logits.shape
        attn_weights = torch.softmax(attention_logits.view(B, K, -1), dim=2).view(
            B, K, H, W
        )

        # --- BƯỚC 3: AGGREGATE CONCEPT VECTORS ---
        # Tạo vector đại diện cho từng bệnh bằng cách nhân Features với Attention Map của bệnh đó
        # Features: [B, C, H*W]
        f_flat = features.view(B, C, -1)
        # Weights:  [B, K, H*W]
        a_flat = attn_weights.view(B, K, -1)

        # Concept Vectors V = Weights x Features^T
        # Concept Vectors: [B, Num_Classes, C]
        concept_vectors = torch.bmm(a_flat, f_flat.permute(0, 2, 1))

        # Projector & Classify
        projected_vectors = self.projector(concept_vectors)
        logits = self.classifier(projected_vectors).squeeze(-1)

        # --- RETURN THÊM projected_vectors ĐỂ TÍNH CONTRASTIVE LOSS ---
        return {
            "logits": logits,
            "attn_maps": attn_weights,
            "concept_vectors": projected_vectors,  # Dùng cái đã qua projector tốt hơn
        }
