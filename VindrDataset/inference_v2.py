import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import các module từ source code của bạn
# Đảm bảo bạn đặt file inference.py cùng cấp với folder src/
from src.model import MedicalConceptModel
from src.dataset import TARGET_CLASSES


def get_args():
    parser = argparse.ArgumentParser(
        description="Inference VinDr-CXR with Saliency Maps"
    )

    # Tham số bắt buộc
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn đến file ảnh (.jpg, .png, .dicom) HOẶC đường dẫn folder chứa ảnh",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Đường dẫn đến file weights (.pth) đã train",
    )

    # Tham số tuỳ chọn
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất để hiển thị bệnh (Default: 0.5)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Kích thước ảnh input cho model (Default: 384)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Thiết bị chạy inference: 'cuda' hoặc 'cpu'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Folder lưu ảnh kết quả (nếu không set, sẽ chỉ in ra màn hình)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Nếu dùng flag này, code sẽ KHÔNG popup cửa sổ ảnh (dùng khi chạy trên server)",
    )

    return parser.parse_args()


def get_val_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def predict_and_visualize(model, image_path, args, transform):
    # 1. Đọc ảnh
    # Xử lý trường hợp ảnh DICOM nếu cần (ở đây demo với ảnh thường jpg/png)
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file: {image_path}")
        return

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ Lỗi đọc ảnh: {image_path}")
        return

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = original_img.shape

    # 2. Preprocess
    augmented = transform(image=original_img)
    input_tensor = augmented["image"].unsqueeze(0).to(args.device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs["logits"][0]
    attn_maps = outputs["attn_maps"][0]  # [Num_Classes, H, W]

    probs = torch.sigmoid(logits).cpu().numpy()

    # 4. Xử lý kết quả
    active_indices = np.where(probs > args.threshold)[0]
    filename = os.path.basename(image_path)

    print(f"\n--- 📸 Kết quả cho: {filename} ---")

    if len(active_indices) == 0:
        print("✅ Kết luận: Bình thường / Không phát hiện bệnh (No findings)")
        # Vẫn lưu ảnh gốc nếu cần
        if args.output_dir:
            save_path = os.path.join(args.output_dir, f"clean_{filename}")
            plt.imsave(save_path, original_img)
        return

    # Nếu có bệnh -> Vẽ Heatmap chồng lên
    for idx in active_indices:
        disease_name = TARGET_CLASSES[idx]
        score = probs[idx]
        print(f"⚠️ Phát hiện: {disease_name} (Score: {score:.2f})")

        # Xử lý Heatmap
        heatmap = attn_maps[idx].cpu().numpy()

        # Normalize về 0-255
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap)

        # Resize về kích thước gốc
        heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))

        # Tạo màu
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Chồng ảnh (Overlay)
        # Chuyển heatmap_color từ BGR sang RGB để hiển thị đúng bằng matplotlib
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

        # 5. Lưu hoặc Hiển thị
        title = f"{disease_name} ({score:.2f})"

        # Lưu ra file
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            # Tên file: tenanh_tenbenh.jpg
            save_name = (
                f"{os.path.splitext(filename)[0]}_{disease_name.replace(' ', '_')}.jpg"
            )
            save_path = os.path.join(args.output_dir, save_name)

            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title(title, fontsize=15, color="red")
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()  # Đóng figure để giải phóng ram
            print(f"   💾 Đã lưu ảnh phân tích tại: {save_path}")

        # Hiển thị popup (nếu không chặn)
        if not args.no_show:
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(title)
            plt.axis("off")
            plt.show()


def main():
    args = get_args()

    # 1. Setup Model
    print(f"⏳ Đang load model từ {args.checkpoint}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Khởi tạo kiến trúc model (phải khớp với lúc train)
    model = MedicalConceptModel(num_classes=len(TARGET_CLASSES))

    # Load weights
    try:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        print("Gợi ý: Kiểm tra lại kiến trúc model hoặc đường dẫn file.")
        return

    model.to(device)
    model.eval()
    print("✅ Load model thành công!")

    # 2. Setup Transform
    val_transform = get_val_transform(args.img_size)

    # 3. Xác định danh sách ảnh cần chạy
    image_paths = []
    if os.path.isdir(args.input):
        # Nếu input là folder, lấy tất cả ảnh jpg/png/jpeg
        types = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for t in types:
            image_paths.extend(glob.glob(os.path.join(args.input, t)))
        print(f"📂 Tìm thấy {len(image_paths)} ảnh trong thư mục.")
    else:
        # Nếu input là file đơn lẻ
        image_paths = [args.input]

    # 4. Chạy vòng lặp dự đoán
    for img_path in image_paths:
        predict_and_visualize(model, img_path, args, val_transform)


if __name__ == "__main__":
    main()
