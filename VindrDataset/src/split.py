import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load dữ liệu gốc
df = pd.read_csv(
    "/home/aaronpham/Coding/SoftAI---DataForLife---MedSightAI/VindrDataset/src/train_with_meta.csv"
)

# 2. Gom nhóm theo Image_ID (Vì 1 ảnh có nhiều dòng)
# Tạo cột 'labels_list' chứa tất cả các bệnh của ảnh đó
df_grouped = df.groupby("image_id")["class_id"].apply(list).reset_index()


# Hàm tạo "Stratify Key" - Mẹo để chia đều
# Chúng ta sẽ ưu tiên chia đều dựa trên bệnh "hiếm nhất" mà ảnh đó có
def get_stratify_group(labels):
    # Nếu ảnh "No finding" (thường là 14), gán nhóm riêng
    if 14 in labels and len(labels) == 1:
        return "Normal"
    # Nếu có bệnh, lấy ID bệnh nhỏ nhất (thường bệnh hiếm có ID thấp hoặc cao tùy dataset,
    # nhưng ở đây ta dùng string kết hợp để phân loại sơ bộ)
    labels = sorted(list(set([l for l in labels if l != 14])))
    if len(labels) == 0:
        return "Normal"
    return str(labels[0])  # Stratify theo bệnh đầu tiên tìm thấy


df_grouped["stratify_col"] = df_grouped["class_id"].apply(get_stratify_group)

# 3. Chia tập Test (15%) - Giữ nguyên tỷ lệ bệnh
train_val, test = train_test_split(
    df_grouped, test_size=0.15, stratify=df_grouped["stratify_col"], random_state=42
)

# 4. Chia Train (70%) / Val (15%) từ phần còn lại
train, val = train_test_split(
    train_val,
    test_size=0.176,  # 0.176 của 85% ~ 15% tổng thể
    stratify=train_val["stratify_col"],
    random_state=42,
)

# 5. Bung ngược lại ra format gốc (Explode) để Dataloader đọc được
# Lọc df gốc chỉ lấy những image_id nằm trong tập train/val/test tương ứng
df_train = df[df["image_id"].isin(train["image_id"])]
df_val = df[df["image_id"].isin(val["image_id"])]
df_test = df[df["image_id"].isin(test["image_id"])]

# 6. Lưu file
df_train.to_csv("train_split.csv", index=False)
df_val.to_csv("val_split.csv", index=False)
df_test.to_csv("test_split.csv", index=False)

print(f"Train size: {len(df_train['image_id'].unique())} ảnh")
print(f"Val size:   {len(df_val['image_id'].unique())} ảnh")
print(f"Test size:  {len(df_test['image_id'].unique())} ảnh")
