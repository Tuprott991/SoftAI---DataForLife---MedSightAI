# 🔴 CRITICAL ISSUES FOUND - Why Your Model Isn't Learning

After analyzing your code, I found **5 CRITICAL ISSUES** that explain the poor performance (AUC ~0.5):

---

## 🔴 ISSUE #1: BBoxGuidedConceptLoss is COMPLETELY BROKEN

**Location**: `utils_bbox.py`, line 62-68

**Current Code**:
```python
# Loss: maximize CAM inside, minimize CAM outside
# Using hinge loss style: want cam > 0 inside, cam < 0 outside
inside_loss = F.relu(-cam * inside_mask).sum() / (inside_mask.sum() + 1e-6)
outside_loss = F.relu(cam * outside_mask).sum() / (outside_mask.sum() + 1e-6)
```

**The Problem**:
- You're penalizing HIGH activations INSIDE the bbox! (`-cam * inside_mask`)
- You're penalizing HIGH activations OUTSIDE the bbox! (`cam * outside_mask`)
- **This is BACKWARDS** - the model learns to predict LOW everywhere!
- Result: CAMs become uniform noise → AUC = 0.5 (random)

**What Should Happen**:
- Inside bbox: CAM should be HIGH (positive after sigmoid)
- Outside bbox: CAM should be LOW (close to 0)

**Fix**:
```python
# Apply sigmoid to get activations in [0, 1]
cam_sigmoid = torch.sigmoid(cam)  # (H, W)

# Inside bbox: want HIGH activation (close to 1)
# Outside bbox: want LOW activation (close to 0)
inside_target = inside_mask  # Want 1 inside
outside_target = 1 - outside_mask  # Want 0 outside (so 1 - mask)

# MSE loss or BCE
inside_loss = ((cam_sigmoid - inside_target) ** 2 * inside_mask).sum() / (inside_mask.sum() + 1e-6)
outside_loss = ((cam_sigmoid - 0) ** 2 * outside_mask).sum() / (outside_mask.sum() + 1e-6)
```

**Impact**: This single bug destroys Stage 1 learning completely!

---

## 🔴 ISSUE #2: Learning Rate WAY TOO LOW for Stage 1

**Location**: `train.py`, line 418-421

**Current Code**:
```python
optimizer = optim.AdamW([
    {'params': model.module.backbone.parameters(), 'lr': args.lr * 0.01},  # 1e-6
    {'params': model.module.concept_head.parameters(), 'lr': args.lr * 0.2}  # 2e-5
])
```

**The Problem**:
- Base LR = 1e-4
- Backbone LR = **1e-6** (extremely low!)
- Concept head LR = **2e-5** (too low!)
- With 50 epochs @ 53 batches = **2,650 update steps only**
- At LR=1e-6, backbone barely moves at all
- MedMAE pretrained weights are good but need fine-tuning

**What Happens**:
- Model can't adapt to your specific concepts
- Concept head can't learn proper attention maps
- AUC stays at 0.5 (random initialization)

**Fix**:
```python
optimizer = optim.AdamW([
    {'params': model.module.backbone.parameters(), 'lr': args.lr * 0.1},  # 1e-5 (10x higher)
    {'params': model.module.concept_head.parameters(), 'lr': args.lr}      # 1e-4 (5x higher)
])
```

**OR use a warmup schedule:**
```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = optim.AdamW([
    {'params': model.module.backbone.parameters(), 'lr': 5e-5},
    {'params': model.module.concept_head.parameters(), 'lr': 5e-4}
])

scheduler = OneCycleLR(optimizer, max_lr=[5e-5, 5e-4], 
                       steps_per_epoch=len(train_loader), 
                       epochs=args.epochs_stage1)
```

---

## 🔴 ISSUE #3: Stage 2 Has NO VALIDATION

**Location**: `train.py`, line 486-504

**Current Code**:
```python
for epoch in range(args.epochs_stage2):
    loss = train_one_epoch(...)
    if rank == 0:
        print(f"Epoch {epoch+1}: Loss {loss:.4f}")
    # NO VALIDATION!!!
```

**The Problem**:
- Stage 2 trains for 50 epochs with **NO METRICS**
- You can't tell if prototypes are learning anything useful
- Loss goes down (7.4 → 6.6) but this doesn't mean good prototypes
- Prototypes might be collapsing or diverging

**Fix**: Add validation with similarity metrics
```python
# After each epoch in Stage 2:
if rank == 0 and epoch % 5 == 0:
    # Check prototype quality
    with torch.no_grad():
        # Compute prototype diversity (should be high)
        prototype_similarity = F.cosine_similarity(
            prototypes.view(-1, 1, 128), 
            prototypes.view(1, -1, 128), dim=2
        )
        # Remove diagonal
        mask = ~torch.eye(len(prototypes), dtype=bool)
        avg_similarity = prototype_similarity[mask].mean()
        print(f"  Prototype avg similarity: {avg_similarity:.4f} (want < 0.5)")
```

---

## 🔴 ISSUE #4: Stage 2 Loss Might Be Wrong

**Location**: `train.py`, line 109-116

**Current Code in train loop**:
```python
# Stage 2: Get local vectors and compute contrastive loss
f_proj = outputs['projected_features']
local_vectors = actual_model.get_local_concept_vectors(f_proj, outputs['cams'])
loss = criterion(local_vectors, actual_model.prototypes, concepts_gt, 
                 num_prototypes_per_concept=actual_model.M)
```

**Potential Issues**:
1. Are you using CAMs from Stage 1 (frozen)? Check if they're detached
2. Is PrototypeContrastiveLoss pulling prototypes together correctly?
3. Temperature=0.1 might be too low (makes loss too sharp)

**Need to verify PrototypeContrastiveLoss implementation** - can you share `utils.py`?

---

## 🔴 ISSUE #5: No Gradient Flow Check

**The Problem**:
- DDP with `find_unused_parameters=True` masks gradient issues
- Gradients might be vanishing or exploding
- No gradient monitoring during training

**Fix**: Add gradient monitoring
```python
# After loss.backward() in train_one_epoch:
if rank == 0 and batch_idx % 10 == 0:
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Grad norm: {total_norm:.4f}")
    if total_norm < 1e-7:
        print(f"  ⚠️ Vanishing gradients!")
    if total_norm > 100:
        print(f"  ⚠️ Exploding gradients!")
```

---

## 📊 Additional Observations

### Stage 1 Metrics Show No Learning:
- AUC fluctuates randomly: 0.50 → 0.53 → 0.48 → 0.46 (pure noise)
- mAP stays ~0.045-0.05 (very low, barely better than random)
- Loss decreases but metrics don't improve → **overfitting to wrong objective**

### Stage 3 Shows Slight Improvement:
- AUC reaches 0.60 (barely above random)
- Model predicting same class for everything:
  - F1-macro = 0.14 (terrible per-class performance)
  - F1-micro = 0.67 (high because predicting majority class)
- This means: **Model learned to predict "No finding" for everything**

---

## 🔧 IMMEDIATE FIXES TO IMPLEMENT

### Priority 1: Fix BBoxGuidedConceptLoss (CRITICAL)
This is THE main bug killing your model.

### Priority 2: Increase Stage 1 Learning Rate
Current LR is 10-100x too low.

### Priority 3: Add Stage 2 Validation
You need to monitor prototype learning.

### Priority 4: Check Data Quality
Run the `check_dataloader.py` script I created to verify:
- Labels are correct
- BBox annotations are valid
- No all-zero or all-one columns
- Class balance isn't too extreme

---

## 🎯 Expected Results After Fixes

**Stage 1** (after fixing bbox loss + LR):
- AUC should reach 0.65-0.75 within 20 epochs
- mAP should reach 0.15-0.25
- Loss should decrease smoothly

**Stage 3** (after proper Stage 1 + 2):
- AUC should reach 0.70-0.80 on test set
- F1-macro should be 0.30+
- Not all predictions should be the same class

---

## 🚀 Next Steps

1. **RUN**: `python check_dataloader.py` to verify data
2. **FIX**: Apply the bbox loss fix (Priority 1)
3. **FIX**: Increase learning rates (Priority 2)
4. **RETRAIN**: With fixed code
5. **MONITOR**: Check if AUC improves above 0.6 in Stage 1
6. **SHARE**: Results and I'll help further

Do you want me to implement these fixes in your code?
