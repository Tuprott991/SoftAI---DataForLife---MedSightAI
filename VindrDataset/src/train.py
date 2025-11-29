import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

# Import các module đã xây dựng
from dataset import VINDRCXRDataset
from model import MedicalConceptModel
from loss import VinDrCompositeLoss


def get_args():
    parser = argparse.ArgumentParser(
        description="Training Script for VinDr-CXR Concept Model"
    )

    # --- 1. PATH PARAMETERS (SỬA ĐỔI) ---
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./train_split.csv",
        help="Đường dẫn file train.csv (Tập Train)",
    )
    parser.add_argument(
        "--val_csv_path",
        type=str,
        default="./val_split.csv",
        help="Đường dẫn file validation.csv",
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default="./test_split.csv",
        help="Đường dẫn file test.csv",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./train_images",
        help="Thư mục chứa ảnh DICOM/PNG",
    )
    parser.add_argument(
        "--save_path", type=str, default="./checkpoints", help="Thư mục lưu model"
    )

    # --- 2. QUICK TEST PARAMETERS ---
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số lượng ảnh để train nhanh (VD: 500). Chỉ áp dụng cho tập Train.",
    )

    # --- 3. HYPERPARAMETERS (Giữ nguyên các giá trị tối ưu) ---
    parser.add_argument("--epochs", type=int, default=30, help="Số lượng epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, default=384, help="Kích thước ảnh đầu vào"
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=384,
        help="Kích thước Attention Map GT (nên bằng img_size)",
    )
    parser.add_argument("--num_classes", type=int, default=14, help="Số lớp bệnh lý")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed để tái lập kết quả"
    )

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_seg = 0.0
    running_con = 0.0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Unpack data
        images, cls_labels, attn_maps, ids = batch

        images = images.to(device)
        cls_labels = cls_labels.to(device)
        attn_maps = attn_maps.to(device)

        targets = {"cls_label": cls_labels, "attn_maps": attn_maps}

        # Forward
        optimizer.zero_grad()
        outputs = model(images)

        # Compute Loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        # Backward
        loss.backward()
        optimizer.step()

        # Logging
        running_loss += loss.item()
        running_cls += loss_dict["loss_cls"].item()
        running_seg += loss_dict["loss_seg"].item()
        running_con += loss_dict["loss_con"].item()

        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Cls": f"{loss_dict['loss_cls'].item():.2f}"}
        )

    num_batches = len(loader)
    return {
        "loss": running_loss / num_batches,
        "cls": running_cls / num_batches,
        "seg": running_seg / num_batches,
        "con": running_con / num_batches,
    }


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, cls_labels, attn_maps, ids = batch

            images = images.to(device)
            cls_labels = cls_labels.to(device)
            attn_maps = attn_maps.to(device)

            targets = {"cls_label": cls_labels, "attn_maps": attn_maps}
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            running_loss += loss_dict["loss"].item()

    return running_loss / len(loader)


def main():
    args = get_args()

    # Tạo thư mục lưu model nếu chưa có
    os.makedirs(args.save_path, exist_ok=True)

    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")

    # --- 1. SETUP DATA (DÙNG 3 FILE SPLIT) ---

    # 1. Khởi tạo 3 Dataset riêng biệt từ 3 file CSV
    print(f"-> Loading training data from {args.csv_path}...")
    train_dataset_full = VINDRCXRDataset(
        csv_file=args.csv_path,
        image_dir=args.image_path,
        num_classes=args.num_classes,
        target_size=args.img_size,
        map_size=args.map_size,
    )
    val_dataset = VINDRCXRDataset(
        csv_file=args.val_csv_path,
        image_dir=args.image_path,
        num_classes=args.num_classes,
        target_size=args.img_size,
        map_size=args.map_size,
    )
    test_dataset = VINDRCXRDataset(
        csv_file=args.test_csv_path,
        image_dir=args.image_path,
        num_classes=args.num_classes,
        target_size=args.img_size,
        map_size=args.map_size,
    )

    # 2. Xử lý QUICK TRAIN (Subset) - Chỉ áp dụng cho tập Train
    if args.max_samples is not None and args.max_samples < len(train_dataset_full):
        print(
            f"-> QUICK TEST MODE: Sử dụng {args.max_samples} ảnh ngẫu nhiên cho Train."
        )
        indices = np.random.choice(
            len(train_dataset_full), args.max_samples, replace=False
        )
        train_set = Subset(train_dataset_full, indices)
    else:
        print(
            f"-> FULL TRAIN MODE: Sử dụng toàn bộ {len(train_dataset_full)} ảnh cho Train."
        )
        train_set = train_dataset_full  # Giờ train_set là tập train đã tách

    # 3. Khởi tạo 3 DataLoader
    # DataLoader Train
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # DataLoader Validation
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(
        f"-> Data sizes: Train={len(train_set)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # --- 2. SETUP MODEL & OPTIMIZER ---
    print("-> Initializing Model...")
    model = MedicalConceptModel(num_classes=args.num_classes, model_name="resnet50")
    model.to(device)

    # Khởi tạo Criterion (truyền num_classes và device để tính pos_weight chính xác)
    criterion = VinDrCompositeLoss(num_classes=args.num_classes, device=device).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- [NEW] 2.1 RESUME LOGIC ---
    start_epoch = 0
    best_loss = float("inf")
    resume_path = os.path.join(args.save_path, "last_model.pth")

    if os.path.exists(resume_path):
        print(f"-> Phát hiện checkpoint cũ tại {resume_path}. Đang load...")
        checkpoint = torch.load(resume_path, map_location=device)

        # Load Model Weight
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)  # Fallback cho file cũ chỉ lưu weight

        # Load Optimizer & Epoch (Quan trọng để train tiếp mượt mà)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]

        print(f"-> Resume thành công! Tiếp tục từ Epoch {start_epoch + 1}")
    else:
        print("-> Không tìm thấy checkpoint cũ. Train từ đầu.")

    # --- 3. TRAIN LOOP ---
    # Chạy từ start_epoch đến args.epochs
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Train Loss: {train_metrics['loss']:.4f} (Cls: {train_metrics['cls']:.3f}, Seg: {train_metrics['seg']:.3f}, Con: {train_metrics['con']:.3f})"
        )

        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Save Best Model (Chỉ lưu weight)
        if val_loss < best_loss:
            best_loss = val_loss
            save_name = os.path.join(args.save_path, "best_model.pth")
            torch.save(model.state_dict(), save_name)
            print(f"-> Saved Best Model to {save_name}")

        # Save Last Model (Full Checkpoint để Resume)
        checkpoint_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint_dict, os.path.join(args.save_path, "last_model.pth"))


if __name__ == "__main__":
    main()
