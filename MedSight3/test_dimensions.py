"""
Quick test to verify all dimensions match correctly.
Run this BEFORE training to catch dimension mismatches early.
"""

import torch
from src.dataloader_bbox_simple import get_dataloaders_with_bbox
import pandas as pd

def test_dimensions(train_csv, test_csv, train_images_dir, test_images_dir, 
                   train_bbox_csv, test_bbox_csv, batch_size=16):
    """Test that all tensor dimensions match correctly."""
    
    print("="*80)
    print("🔍 TESTING TENSOR DIMENSIONS")
    print("="*80)
    
    # 1. Load dataloaders
    print("\n1️⃣ Loading dataloaders with filtering...")
    train_loader, val_loader, test_loader, num_concepts, num_classes, _ = get_dataloaders_with_bbox(
        train_csv, test_csv, train_images_dir, test_images_dir,
        train_bbox_csv, test_bbox_csv,
        batch_size=batch_size,
        rank=0, world_size=1,
        val_split=0.1,
        filter_rare=True,
        balance_no_finding=True
    )
    
    print(f"✅ Dataloaders created: {num_concepts} concepts, {num_classes} classes")
    
    # 2. Get filtered columns from dataloader
    concept_cols = train_loader.dataset.concept_cols
    target_cols = train_loader.dataset.target_cols
    
    print(f"✅ Concept columns ({len(concept_cols)}): {concept_cols[:3]}... (showing first 3)")
    print(f"✅ Target columns ({len(target_cols)}): {target_cols}")
    
    # 3. Verify dimensions match
    assert len(concept_cols) == num_concepts, \
        f"❌ Mismatch! len(concept_cols)={len(concept_cols)} != num_concepts={num_concepts}"
    assert len(target_cols) == num_classes, \
        f"❌ Mismatch! len(target_cols)={len(target_cols)} != num_classes={num_classes}"
    
    print(f"✅ Column counts match dataloader outputs")
    
    # 4. Test pos_weight computation (Stage 1)
    print("\n2️⃣ Testing pos_weight computation (Stage 1)...")
    df_train = pd.read_csv(train_csv)
    df_agg = df_train.groupby('image_id')[concept_cols + target_cols].max().reset_index()
    
    concept_values = torch.tensor(df_agg[concept_cols].values, dtype=torch.float32)
    pos_weight = (concept_values == 0).sum(dim=0) / (concept_values == 1).sum(dim=0).clamp(min=1)
    pos_weight = pos_weight.clamp(max=20.0)
    
    print(f"✅ pos_weight shape: {pos_weight.shape}")
    print(f"✅ Expected shape: torch.Size([{num_concepts}])")
    
    assert pos_weight.shape[0] == num_concepts, \
        f"❌ pos_weight dimension mismatch! {pos_weight.shape[0]} != {num_concepts}"
    
    print(f"✅ pos_weight dimensions correct!")
    
    # 5. Test class distribution computation (Stage 3)
    print("\n3️⃣ Testing class distribution computation (Stage 3)...")
    target_values = torch.tensor(df_agg[target_cols].values, dtype=torch.float32)
    samples_per_class = target_values.sum(dim=0)
    
    print(f"✅ samples_per_class shape: {samples_per_class.shape}")
    print(f"✅ Expected shape: torch.Size([{num_classes}])")
    
    assert samples_per_class.shape[0] == num_classes, \
        f"❌ samples_per_class dimension mismatch! {samples_per_class.shape[0]} != {num_classes}"
    
    print(f"✅ Stage 3 class distribution dimensions correct!")
    
    # 6. Test a batch
    print("\n4️⃣ Testing batch dimensions...")
    batch = next(iter(train_loader))
    images, concepts, targets, bboxes, has_bbox = batch
    
    print(f"✅ Batch shapes:")
    print(f"  - images: {images.shape}")
    print(f"  - concepts: {concepts.shape} (expected: [batch, {num_concepts}])")
    print(f"  - targets: {targets.shape} (expected: [batch, {num_classes}])")
    print(f"  - bboxes: {bboxes.shape}")
    print(f"  - has_bbox: {has_bbox.shape}")
    
    assert concepts.shape[1] == num_concepts, \
        f"❌ Batch concept dimension mismatch! {concepts.shape[1]} != {num_concepts}"
    assert targets.shape[1] == num_classes, \
        f"❌ Batch target dimension mismatch! {targets.shape[1]} != {num_classes}"
    
    print(f"✅ Batch dimensions correct!")
    
    # 7. Summary
    print("\n" + "="*80)
    print("✅ ALL DIMENSION CHECKS PASSED!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - num_concepts: {num_concepts}")
    print(f"  - num_classes: {num_classes}")
    print(f"  - pos_weight shape: {pos_weight.shape[0]}")
    print(f"  - samples_per_class shape: {samples_per_class.shape[0]}")
    print(f"  - batch concepts shape: {concepts.shape[1]}")
    print(f"  - batch targets shape: {targets.shape[1]}")
    print(f"\n✅ All dimensions are consistent. Safe to train!")
    print("="*80)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dimensions before training')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--train_images_dir', type=str, required=True)
    parser.add_argument('--test_images_dir', type=str, required=True)
    parser.add_argument('--train_bbox_csv', type=str, required=True)
    parser.add_argument('--test_bbox_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    try:
        test_dimensions(
            args.train_csv, args.test_csv,
            args.train_images_dir, args.test_images_dir,
            args.train_bbox_csv, args.test_bbox_csv,
            args.batch_size
        )
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
