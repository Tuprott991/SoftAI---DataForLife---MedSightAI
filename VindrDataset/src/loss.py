import torch
import torch.nn as nn
import torch.nn.functional as F


class VinDrCompositeLoss(nn.Module):
    def __init__(
        self,
        num_classes=14,
        w_cls=1.0,
        w_seg=5.0,
        w_con=1.0,
        pos_weight_value=10.0,
        device="cuda",
    ):
        """
        Args:
            pos_weight_value (float): Trọng số phạt cho ca dương tính (Giải quyết lỗi 0.23).
                                      Nên set khoảng 10.0 đến 20.0 cho VinDr.
        """
        super().__init__()
        self.w_cls = w_cls
        self.w_seg = w_seg
        self.w_con = w_con

        # --- 1. GIẢI QUYẾT IMBALANCE (Lỗi Prob 0.23) ---
        # Tạo vector trọng số: Nếu bỏ sót bệnh (Positive), phạt nặng gấp 10 lần bỏ sót ca thường.
        # Lưu ý: Cần đưa pos_weight vào đúng device (GPU/CPU)
        pos_weight = torch.tensor([pos_weight_value] * num_classes).to(device)
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Loss cho Attention Map (Mask)
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def contrastive_loss(self, features, labels, temperature=0.07):
        """
        Tính Contrastive Loss.
        Sửa lỗi: Đảm bảo luôn trả về Tensor để tránh lỗi .item() ở Epoch 8
        """
        batch_size = features.shape[0]
        num_classes = features.shape[1]

        # --- 2. GIẢI QUYẾT CRASH EPOCH 8 ---
        # SAI: total_loss = 0.0 (Float gây lỗi attribute error)
        # ĐÚNG: Khởi tạo là một Tensor số 0
        total_loss = torch.tensor(0.0, device=features.device)

        for k in range(num_classes):
            vecs = features[:, k, :]  # [B, Dim]
            lbls = labels[:, k]  # [B]

            # Chỉ tính nếu batch có cả Positive và Negative
            if lbls.sum() > 0 and lbls.sum() < batch_size:
                # Cosine similarity matrix
                sim_matrix = torch.matmul(vecs, vecs.T) / temperature

                # Mask: 1 nếu cùng nhãn
                pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).float()

                # Bỏ qua đường chéo (chính nó)
                mask = pos_mask * (1 - torch.eye(batch_size, device=features.device))

                # InfoNCE calculation
                exp_sim = torch.exp(sim_matrix)
                # Tránh log(0) bằng cách cộng 1e-6
                log_prob = sim_matrix - torch.log(
                    exp_sim.sum(dim=1, keepdim=True) + 1e-6
                )

                # Mean loss over positive pairs
                # Tránh chia cho 0 nếu mask.sum = 0
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

                loss = -mean_log_prob_pos.mean()

                # Cộng dồn loss (Tensor + Tensor)
                total_loss = total_loss + loss

        return total_loss / num_classes

    def forward(self, outputs, targets):
        # Unpack outputs từ Model
        logits = outputs["logits"]  # [B, K]
        pred_maps_logits = outputs["attn_maps"]  # [B, K, h, w]
        concept_vecs = outputs.get("concept_vectors", None)

        # Unpack targets từ Dataset
        gt_labels = targets["cls_label"]  # [B, K]
        gt_maps = targets["attn_maps"]  # [B, K, H, W]

        # 1. Classification Loss (Đã có pos_weight)
        loss_cls = self.cls_criterion(logits, gt_labels)

        # 2. Segmentation Loss
        # Resize output map của model lên kích thước GT mask
        if pred_maps_logits.shape[-2:] != gt_maps.shape[-2:]:
            pred_maps_up = F.interpolate(
                pred_maps_logits,
                size=gt_maps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            pred_maps_up = pred_maps_logits

        loss_seg = self.seg_criterion(pred_maps_up, gt_maps)

        # 3. Contrastive Loss
        # Kiểm tra concept_vecs có tồn tại không
        if concept_vecs is not None:
            loss_con = self.contrastive_loss(concept_vecs, gt_labels)
        else:
            # Fallback an toàn cũng phải là Tensor
            loss_con = torch.tensor(0.0, device=logits.device)

        # Tổng hợp Loss
        total_loss = (
            (self.w_cls * loss_cls) + (self.w_seg * loss_seg) + (self.w_con * loss_con)
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_seg": loss_seg,
            "loss_con": loss_con,
        }
