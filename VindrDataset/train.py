import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler  # Thư viện để train mixed precision


# IMPORT MODULE CỦA BẠN
from src.dataset import VinDrClassifierDataset, TARGET_CLASSES
from src.model import MedicalConceptModel
from src.loss import VinDrLoss

# --- CẤU HÌNH ---
CONFIG = {
    # Thay đổi đường dẫn tới folder VinDr của bạn
    "csv_file": "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv",
    "image_dir": "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/",
    "batch_size": 16,  # Tăng giảm tuỳ VRAM (16 với T4/P100 là ổn)
    "lr": 1e-4,  # Learning rate
    "epochs": 15,  # VinDr cần khoảng 10-20 epochs để hội tụ tốt
    "img_size": 384,  # Kích thước input chuẩn của SigLIP
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "./checkpoints",
}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_con = 0.0
    running_reg = 0.0

    pbar = tqdm(loader, desc="Training")

    for images, labels, masks in pbar:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)  # Đẩy mask lên GPU

        with autocast(enabled=(scaler is not None)):
            # Trường hợp A: Model cần mask (ít gặp ở CNN thuần, hay gặp ở Transformer)
            # outputs = model(images, masks)

            # Trường hợp B: Model chỉ cần ảnh
            outputs = model(images)

            # SỬA 2: Truyền mask vào hàm loss (nếu hàm loss yêu cầu)
            # Ví dụ: criterion(outputs, labels, masks)
            loss_dict = criterion(outputs, labels, masks)
            loss = loss_dict["total_loss"]

        # 2. Backward & Optimize
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 3. Logging
        running_loss += loss.item()

        # Log riêng từng phần để debug
        # Nếu loss cls giảm mà loss con không giảm -> cần chỉnh weight
        if "loss_cls" in loss_dict:
            running_cls += loss_dict["loss_cls"].item()
        if "loss_con" in loss_dict:
            running_con += loss_dict["loss_con"].item()
        if "loss_reg" in loss_dict:
            running_reg += loss_dict["loss_reg"].item()

        # Hiển thị trên thanh progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Cls": f"{loss_dict.get('loss_cls', 0):.3f}",
                "Con": f"{loss_dict.get('loss_con', 0):.3f}",
                "Reg": f"{loss_dict.get('loss_reg', 0):.3f}",
            }
        )

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    # Dùng list để lưu prediction tính metric sau này nếu cần (như AUC)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Lưu sigmoid probabilities để tính accuracy/AUC
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    avg_loss = running_loss / len(loader)

    # Tính Accuracy đơn giản (Threshold 0.5)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # Correct nếu: (pred > 0.5) == label
    predicted_labels = (all_preds > 0.5).float()
    accuracy = (predicted_labels == all_targets).float().mean().item()

    return avg_loss, accuracy


def main():
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    print(f"Training on: {CONFIG['device']}")

    # --- 1. SETUP DATA ---
    train_transform = A.Compose(
        [
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # SigLIP normalize
            ToTensorV2(),
        ]
    )

    # Dataset (Loader đã viết ở bước trước)
    full_dataset = VinDrClassifierDataset(
        CONFIG["csv_file"], CONFIG["image_dir"], transform=train_transform
    )

    # --- CHẾ ĐỘ DEBUG/TEST NHANH ---
    DEBUG_MODE = True

    if DEBUG_MODE:
        print("⚠️ ĐANG CHẠY CHẾ ĐỘ DEBUG: Chỉ dùng 2000 ảnh!")
        # Lấy ngẫu nhiên 2000 indices
        indices = torch.randperm(len(full_dataset))[:2000]
        subset_data = torch.utils.data.Subset(full_dataset, indices)

        # Chia train/val trên tập nhỏ này
        train_size = int(0.8 * len(subset_data))
        val_size = len(subset_data) - train_size
        train_set, val_set = torch.utils.data.random_split(
            subset_data, [train_size, val_size]
        )
    else:
        # Chạy thật (Full data)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    # DataLoader
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=2
    )  # Batch to lên
    val_loader = DataLoader(
        val_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
    )

    print(f"Dataset Size: Train={len(train_set)}, Val={len(val_set)}")

    # --- 2. SETUP MODEL ---
    model = MedicalConceptModel(num_classes=len(TARGET_CLASSES))
    model.to(CONFIG["device"])

    # --- 3. HANDLING IMBALANCE (Tuỳ chọn nâng cao) ---
    # Tính toán pos_weight dựa trên tỉ lệ dương tính trong tập train
    # Ở đây để đơn giản ta gán trọng số 1.0, nhưng với VinDr nên tăng trọng số cho bệnh hiếm
    # pos_weights = [10.0] * len(TARGET_CLASSES) # Ví dụ: Coi trọng bệnh gấp 10 lần background
    criterion = VinDrLoss(pos_weights=None, device=CONFIG["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # Scheduler để giảm LR khi loss không giảm nữa
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    # --- 4. TRAIN LOOP ---
    best_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"]
        )
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG["device"])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Avg Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(CONFIG["save_path"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model to {save_path}")

    print("Training Complete!")


if __name__ == "__main__":
    main()
