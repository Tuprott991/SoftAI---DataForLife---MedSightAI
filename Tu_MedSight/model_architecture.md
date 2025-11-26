# Medical Concept Model Architecture

## Tổng quan
Model này là một **Concept-based Interpretable Model** cho phân loại hình ảnh y tế (X-quang ngực), kết hợp:
- **Concept Activation Maps (CAMs)** để xác định vùng quan trọng
- **Prototype Learning** với contrastive loss để học các đại diện concept
- **Multi-prototype per concept** để capture sự đa dạng trong mỗi concept

---

## Kiến trúc chi tiết

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT IMAGE                                        │
│                       (B, 3, 448, 448)                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BACKBONE (Vision Transformer)                           │
│                   (e.g., medsiglip-448-vindr-bin)                           │
│                         [Frozen or Fine-tuned]                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
                        FEATURE MAP (B, C, H, W)
                         C = hidden_size
                         H = W = √(num_patches)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
    ┌───────────────────────────┐    ┌──────────────────────────┐
    │   CONCEPT HEAD            │    │   PROJECTOR P            │
    │   Conv2d(C, K, 1x1)       │    │   Linear(C → 128)        │
    │   No Activation           │    │   ReLU                   │
    └───────────┬───────────────┘    │   Linear(128 → 128)      │
                │                     │   L2-Normalize           │
                ▼                     └──────────┬───────────────┘
        RAW CAMs (B, K, H, W)                   │
                │                               ▼
                │                    PROJECTED PATCHES (B, S, P)
                │                    S = H×W, P = projection_dim
                │                               │
                ▼                               │
    ┌──────────────────────────────┐           │
    │  SPATIAL SOFTMAX             │           │
    │  Normalize over (H, W)       │           │
    └──────────┬───────────────────┘           │
               │                                │
               ▼                                │
    CAM WEIGHTS (B, K, H, W)                    │
               │                                │
               │                                │
    ┌──────────┴────────────────────┐          │
    │  Weighted Sum:                │          │
    │  v_local[k] = Σ(w[k,h,w] ×   │          │
    │              features[h,w])   │          │
    └──────────┬────────────────────┘          │
               │                                │
               ▼                                │
    LOCAL CONCEPT VECTORS (B, K, C)            │
               │                                │
               │ Project & Normalize            │
               ▼                                │
    v_local_proj (B, K, P)                      │
               │                                │
               │                                │
    ┌──────────┴────────────────────────────────┴────────────┐
    │                                                         │
    │           PROTOTYPE MATCHING & SIMILARITY               │
    │                                                         │
    │  Prototypes: (K, M, P) - Learnable & L2-normalized     │
    │  K = num_concepts, M = prototypes_per_concept          │
    │                                                         │
    │  For each patch:                                        │
    │    sim[k,m,h,w] = <patch_proj[h,w], prototype[k,m]>   │
    │                                                         │
    │  For each local concept vector:                         │
    │    dots[k,m] = <v_local_proj[k], prototype[k,m]>      │
    │    q[k,m] = softmax_m(γ × dots[k,m])   (assignment)   │
    │    sim_k = Σ_m (q[k,m] × dots[k,m])                   │
    │                                                         │
    └──────────┬──────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  SIMILARITY MAPS             │
    │  (B, K, M, H, W)            │
    │                              │
    │  Max Pooling over space:     │
    │  skm_max = max_{h,w}(sim)   │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  AGGREGATED SIMILARITY       │
    │  sim_k_agg (B, K)           │
    │  Per-concept scores          │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  CLASSIFIER                  │
    │  Linear(K → num_classes)    │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  CLASS LOGITS                │
    │  (B, num_classes)           │
    └─────────────────────────────┘
```

---

## Các thành phần chính

### 1. **Backbone (Feature Extractor)**
- **Input**: RGB image (3 × 448 × 448)
- **Model**: Vision Transformer (e.g., MedSigLIP)
- **Output**: Feature map (B, C, H, W)
  - C = hidden_size (e.g., 768, 1024)
  - H = W = √(number_of_patches)
- **Freeze Option**: Có thể đóng băng backbone để giữ nguyên pretrained weights

### 2. **Concept Head (CAM Generator)**
- **Architecture**: 1×1 Convolution
- **Input**: Feature map (B, C, H, W)
- **Output**: Raw CAMs (B, K, H, W)
  - K = số lượng concepts (e.g., 12)
  - Không sử dụng activation function
- **Purpose**: Tạo K activation maps, mỗi map tương ứng với 1 concept y tế

### 3. **Spatial Softmax & Local Concept Vectors**
- **Spatial Softmax**: Normalize CAMs theo không gian (H, W)
  - `weights[k] = softmax(CAMs[k])` qua (H×W)
- **Weighted Aggregation**: 
  - `v_local[k] = Σ_{h,w} weights[k,h,w] × features[h,w]`
- **Output**: Local concept vector (B, K, C) - đại diện cho mỗi concept

### 4. **Projector P**
- **Architecture**: 2-layer MLP với ReLU
  - Linear(C → P) → ReLU → Linear(P → P)
  - P = projection_dim (e.g., 128)
- **Input**: Feature vectors (C-dimensional)
- **Output**: Projected embeddings (P-dimensional, L2-normalized)
- **Purpose**: 
  - Project local concept vectors: v_local → v_local_proj
  - Project patch features: patches → patches_proj

### 5. **Learnable Prototypes**
- **Shape**: (K, M, P)
  - K = num_concepts
  - M = prototypes_per_concept (e.g., 4)
  - P = projection_dim
- **Initialization**: Random với L2-normalization
- **Learning**: Cập nhật qua contrastive loss
- **Purpose**: Mỗi concept có M prototypes để capture các biến thể khác nhau

### 6. **Similarity Computation**

#### a. **Patch-wise Similarity Maps**
```
sim_maps[b,k,m,h,w] = <patches_proj[b,h,w], prototypes[k,m]>
skm_max[b,k,m] = max_{h,w}(sim_maps[b,k,m,h,w])
```

#### b. **Concept-level Similarity (với soft assignment)**
```
dots[k,m] = <v_local_proj[k], prototypes[k,m]>
q[k,m] = softmax_m(γ × dots[k,m])    # γ = contrastive_gamma
sim_k = Σ_m (q[k,m] × dots[k,m])     # aggregated per concept
```

### 7. **Contrastive Loss (SoftTriple-inspired)**

**Mục đích**: Học prototypes sao cho:
- Concept vectors gần với prototypes của concept đúng
- Xa với prototypes của các concepts khác

**Công thức**:
```
L = -log( exp(λ(sim_pos + δ)) / Σ_{k'} exp(λ·sim_{k'}) )
```

Trong đó:
- `sim_pos`: similarity với positive concept (đúng)
- `λ`: sharpening factor (contrastive_lambda = 10.0)
- `δ`: margin (contrastive_delta = 0.1)
- `γ`: assignment softmax temperature (contrastive_gamma = 10.0)

**Cơ chế**:
1. Với mỗi concept vector v', tính similarity với TẤT CẢ prototypes
2. Positive = prototypes của concept đúng
3. Maximize sim với positive, minimize với negatives
4. Soft assignment q_m cho phép nhiều prototypes contribute

### 8. **Classifier**
- **Architecture**: Linear layer
- **Input**: sim_k_agg (B, K) - aggregated similarity per concept
- **Output**: class_logits (B, num_classes)
- **Purpose**: Predict final classification từ concept similarities

---

## Luồng dữ liệu (Forward Pass)

```
Image (B,3,448,448)
    ↓
Backbone
    ↓
Feature Map (B,C,H,W)
    ├──→ Concept Head → CAMs (B,K,H,W)
    │       ↓
    │   Spatial Softmax → weights (B,K,H,W)
    │       ↓
    │   Weighted Sum → v_local (B,K,C)
    │       ↓
    │   Projector → v_local_proj (B,K,P)
    │       ↓
    │   Similarity with Prototypes → sim_k_agg (B,K)
    │       ↓
    │   Classifier → class_logits (B,num_classes)
    │
    └──→ Projector → patches_proj (B,S,P)
            ↓
        Similarity Maps with Prototypes → sim_maps (B,K,M,H,W)
            ↓
        Max Pooling → skm_max (B,K,M)
```

---

## Hyperparameters

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `num_concepts` | 12 | Số lượng concepts y tế (e.g., phổi, tim, xương sườn) |
| `num_classes` | 1 | Số lượng classes cần phân loại |
| `projection_dim` | 128 | Dimensionality của projected space |
| `prototypes_per_concept` | 4 | Số prototypes cho mỗi concept |
| `contrastive_lambda` | 10.0 | Sharpening factor cho loss |
| `contrastive_gamma` | 10.0 | Temperature cho soft assignment |
| `contrastive_delta` | 0.1 | Margin cho contrastive loss |
| `freeze_backbone` | True | Đóng băng backbone hay fine-tune |

---

## Output Dictionary

Model trả về dictionary chứa:

```python
{
    "feature_map": (B, C, H, W),           # Backbone features
    "cams": (B, K, H, W),                  # Raw CAMs
    "cam_weights": (B, K, H, W),           # Spatial softmax weights
    "v_local": (B, K, C),                  # Local concept vectors (raw)
    "v_local_proj": (B, K, P),             # Projected local vectors
    "patches_proj": (B, S, P),             # Projected patch features
    "sim_maps": (B, K, M, H, W),           # Similarity maps per prototype
    "skm_max": (B, K, M),                  # Max similarity per prototype
    "sim_k_agg": (B, K),                   # Aggregated per-concept similarity
    "q_assign": (B, K, M),                 # Soft assignment probabilities
    "class_logits": (B, num_classes),      # Final classification logits
    "prototypes": (K, M, P),               # Learnable prototypes
}
```

---

## Ưu điểm của kiến trúc này

1. **Interpretability**: 
   - CAMs cho thấy vùng nào model tập trung vào
   - Prototypes có thể visualize để hiểu model học được gì
   
2. **Multi-prototype Learning**:
   - Mỗi concept có nhiều prototypes → capture sự đa dạng
   - VD: "phổi bất thường" có thể có nhiều dạng khác nhau
   
3. **Contrastive Learning**:
   - Loss function giúp học representation tốt hơn
   - Soft assignment linh hoạt hơn hard assignment
   
4. **Modular Design**:
   - Có thể thay đổi backbone dễ dàng
   - Có thể điều chỉnh số concepts, prototypes
   
5. **Medical Domain Specific**:
   - Pretrained backbone trên medical images
   - Concept-based phù hợp với tư duy của bác sĩ

---

## Use Cases

- **Phát hiện bệnh trên X-quang ngực**: phân loại bình thường/bất thường
- **Phân tích đa bệnh lý**: mỗi concept = 1 vùng/dấu hiệu bệnh lý
- **Hỗ trợ chẩn đoán**: visualize CAMs và prototypes giúp bác sĩ hiểu quyết định của AI
- **Giáo dục y khoa**: minh họa các patterns bệnh lý qua prototypes

