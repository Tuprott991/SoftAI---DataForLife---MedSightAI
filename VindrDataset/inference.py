import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Import các module của bạn (đảm bảo cấu trúc thư mục đúng)
from src.model import MedicalConceptModel
from src.dataset import VINDRCXRDataset

# --- CẤU HÌNH MẶC ĐỊNH ---
DEFAULT_CONFIG = {
    "num_classes": 14,
    "image_size": 384,
    "map_size": 384,  # Sử dụng 384 để khớp với GT Map đã train
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_names": [
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Enlarged PA",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Pleural effusion",
        "Pleural thickening",
    ],
}


def get_args():
    """Hàm phân tích tham số đầu vào từ dòng lệnh"""
    parser = argparse.ArgumentParser(
        description="Inference tool for Medical Concept Model (VinDr-CXR)"
    )

    # Tham số bắt buộc/quan trọng
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Đường dẫn đến file ảnh cần test (PNG/JPG)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Đường dẫn đến file trọng số model (.pth)",
    )

    # Tham số tùy chọn
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train_split.csv",  # Đổi sang file đã split
        help="File CSV train dùng để tính Prototypes",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./train_images_png",  # Trỏ đến folder PNG
        help="Thư mục ảnh train dùng để tính Prototypes",
    )
    parser.add_argument(
        "--prototypes_file",
        type=str,
        default="prototypes.pt",
        help="Đường dẫn lưu/load file vectors mẫu",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất để coi là dương tính",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Số lượng bệnh hiển thị trong kết quả"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Thiết bị chạy (cuda/cpu)",
    )
    parser.add_argument(
        "--recompute_proto",
        action="store_true",
        help="Cờ: Bắt buộc tính toán lại Prototypes dù đã có file lưu",
    )

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Tải model và trọng số đã train"""
    print(f"-> Đang tải model từ {checkpoint_path}...")
    model = MedicalConceptModel(
        num_classes=DEFAULT_CONFIG["num_classes"], model_name="resnet50"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def compute_prototypes(model, csv_file, image_dir, device, save_path):
    """
    Tính vector đặc trưng trung bình (Prototype) cho từng loại bệnh từ tập dữ liệu hỗ trợ.
    Lưu ý: Hàm này dùng VINDRCXRDataset, cần đảm bảo Dataset đang dùng logic đọc PNG
    """
    print(
        "-> Đang tính toán Prototypes từ dữ liệu tham chiếu (việc này có thể mất chút thời gian)..."
    )

    # Tái sử dụng Dataset của bạn (đã được sửa logic đọc PNG)
    dataset = VINDRCXRDataset(
        csv_file=csv_file,
        image_dir=image_dir,  # Dùng folder PNG
        num_classes=DEFAULT_CONFIG["num_classes"],
        target_size=DEFAULT_CONFIG["image_size"],
        map_size=DEFAULT_CONFIG["map_size"],
    )

    subset_indices = np.random.choice(
        len(dataset), size=min(500, len(dataset)), replace=False
    )
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=16, shuffle=False, num_workers=2
    )

    model.eval()

    proto_sums = {
        k: torch.zeros(128).to(device) for k in range(DEFAULT_CONFIG["num_classes"])
    }
    proto_counts = {k: 0 for k in range(DEFAULT_CONFIG["num_classes"])}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Prototypes"):
            images, labels, _, _ = batch
            if images is None:
                continue  # Bỏ qua lỗi

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            concept_vectors = outputs["concept_vectors"]

            for i in range(images.size(0)):
                for k in range(DEFAULT_CONFIG["num_classes"]):
                    if labels[i, k] == 1:
                        proto_sums[k] += concept_vectors[i, k, :]
                        proto_counts[k] += 1

    final_prototypes = {}
    for k in range(DEFAULT_CONFIG["num_classes"]):
        if proto_counts[k] > 0:
            avg_vec = proto_sums[k] / proto_counts[k]
            final_prototypes[k] = F.normalize(avg_vec, p=2, dim=0)
        else:
            final_prototypes[k] = torch.zeros(128).to(device)

    torch.save(final_prototypes, save_path)
    print(f"-> Đã lưu Prototypes vào {save_path}")
    return final_prototypes


def preprocess_image(image_path, target_size):
    """
    ⭐️ SỬA ĐỔI: Đọc và xử lý file PNG/JPG (tối ưu hóa cho tốc độ)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    # Đọc ảnh Grayscale bằng OpenCV
    # Sẽ đọc cả PNG và JPG, bỏ qua DICOM
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Không thể đọc file ảnh hoặc định dạng không hỗ trợ: {image_path}"
        )

    # Resize về kích thước model đã train
    image = cv2.resize(image, (target_size, target_size))

    # Normalize 0-255 -> 0.0-1.0
    img_np = image.astype(np.float32) / 255.0

    # Tạo tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    # Trả về tensor cho model và numpy array (cho visualization)
    return img_tensor, img_np


def visualize_inference(original_img, probs, attn_maps, similarities, args):
    """Vẽ kết quả gồm: Ảnh gốc + Text dự đoán + Heatmap bệnh"""
    # Logic giữ nguyên
    class_names = DEFAULT_CONFIG["class_names"]

    top_indices = np.argsort(probs)[::-1][: args.top_k]

    plt.figure(figsize=(16, 6))

    # --- 1. Ảnh gốc & Thông tin ---
    plt.subplot(1, args.top_k + 1, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title("Input X-Ray")
    plt.axis("off")

    text_info = "Top Predictions:\n"
    for idx in top_indices:
        cls_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        prob_score = probs[idx]
        sim_score = similarities[idx] if similarities else 0.0

        mark = "(!)" if prob_score > args.threshold else ""
        text_info += (
            f"{cls_name}: {prob_score*100:.1f}% | Sim: {sim_score:.2f} {mark}\n"
        )

    plt.xlabel(text_info, fontsize=11, loc="left")

    # --- 2. Heatmaps của Top K bệnh ---
    for i, idx in enumerate(top_indices):
        cls_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"

        heatmap = attn_maps[idx].cpu().numpy()

        # Resize heatmap lên kích thước ảnh gốc (384x384)
        heatmap_resized = cv2.resize(
            heatmap,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
            heatmap_resized.max() - heatmap_resized.min() + 1e-8
        )
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        img_rgb = cv2.cvtColor(
            (original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

        plt.subplot(1, args.top_k + 1, i + 2)
        plt.imshow(overlay)
        plt.title(f"{cls_name}\nProb: {probs[idx]:.2f} / Sim: {similarities[idx]:.2f}")
        plt.axis("off")

    plt.tight_layout()
    save_path = "result_" + os.path.basename(args.image_path).split(".")[0] + ".png"
    plt.savefig(save_path)
    print(f"-> Kết quả đã được lưu tại: {save_path}")
    plt.show()


def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Load Model
    model = load_model(args.checkpoint, device)

    # 2. Xử lý Prototypes (cho Cosine Similarity)
    prototypes = {}
    if os.path.exists(args.prototypes_file) and not args.recompute_proto:
        print(f"-> Đang tải Prototypes từ file {args.prototypes_file}...")
        prototypes = torch.load(args.prototypes_file, map_location=device)
    else:
        # Nếu chưa có file hoặc muốn tính lại thì chạy hàm compute
        if os.path.exists(args.train_csv) and os.path.exists(args.train_dir):
            prototypes = compute_prototypes(
                model, args.train_csv, args.train_dir, device, args.prototypes_file
            )
        else:
            print(
                "CẢNH BÁO: Không tìm thấy dữ liệu train để tính Prototypes. Chức năng Similarity sẽ bị tắt."
            )

    # 3. Inference trên ảnh input
    img_tensor, original_img = preprocess_image(
        args.image_path, DEFAULT_CONFIG["image_size"]
    )
    img_tensor = img_tensor.to(device)

    print(f"-> Đang dự đoán cho ảnh: {args.image_path}")
    with torch.no_grad():
        outputs = model(img_tensor)

    # Lấy kết quả
    logits = outputs["logits"][0]
    probs = torch.sigmoid(logits).cpu().numpy()
    attn_maps = outputs["attn_maps"][0]
    concept_vecs = outputs["concept_vectors"][0]

    # Tính Cosine Similarity
    similarities = {}
    if prototypes:
        for k, proto_vec in prototypes.items():
            sim = F.cosine_similarity(concept_vecs[k], proto_vec, dim=0)
            similarities[k] = sim.item()
    else:
        similarities = {k: 0.0 for k in range(DEFAULT_CONFIG["num_classes"])}

    # 4. Hiển thị
    visualize_inference(original_img, probs, attn_maps, similarities, args)


if __name__ == "__main__":
    main()
