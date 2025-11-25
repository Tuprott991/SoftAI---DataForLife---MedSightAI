import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [Batch, Num_Classes, Feature_Dim]
        # labels: [Batch, Num_Classes]
        batch_size, num_classes, feat_dim = features.shape
        total_loss = 0.0

        for i in range(num_classes):
            concept_feats = features[:, i, :]
            concept_feats = F.normalize(concept_feats, dim=1)
            concept_labels = labels[:, i]

            if concept_labels.sum() == 0 or concept_labels.sum() == batch_size:
                continue

            similarity_matrix = torch.matmul(concept_feats, concept_feats.T)
            label_matrix = concept_labels.unsqueeze(0) == concept_labels.unsqueeze(1)
            mask = label_matrix.float().to(features.device)

            logits_mask = torch.ones_like(mask) - torch.eye(
                batch_size, device=features.device
            )
            mask = mask * logits_mask

            logits = similarity_matrix / self.temperature
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            # Tránh chia cho 0
            denom = mask.sum(1) + 1e-6
            mean_log_prob_pos = (mask * log_prob).sum(1) / denom

            loss = -mean_log_prob_pos.mean()
            total_loss += loss

        return total_loss / num_classes


class VinDrLoss(nn.Module):
    def __init__(
        self,
        pos_weights=None,
        device="cuda",
        seg_weight=5.0,  # Trọng số cho Segmentation (Bbox)
        contrastive_weight=0.1,
    ):  # Trọng số cho Contrastive
        super().__init__()
        self.seg_weight = seg_weight
        self.contrastive_weight = contrastive_weight

        # 1. Classification Loss (BCE)
        if pos_weights is not None:
            self.cls_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weights).to(device)
            )
        else:
            self.cls_criterion = nn.BCEWithLogitsLoss()

        # 2. Contrastive Loss (Supervised)
        self.con_criterion = ContrastiveLoss(temperature=0.1)

        # 3. Segmentation Loss (Bbox Guided)
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        outputs: dict {'logits', 'attn_maps', 'concept_vectors'}
        targets: dict {'labels', 'masks'}
        """
        # --- UNPACK OUTPUTS ---
        logits = outputs["logits"]
        attn_maps = outputs["attn_maps"]  # [Batch, 26, 24, 24]
        concept_vectors = outputs[
            "concept_vectors"
        ]  # [Batch, 26, 768] (Bắt buộc phải có để tính Contrastive)

        # --- UNPACK TARGETS ---
        gt_labels = targets["labels"]  # [Batch, 26]
        gt_masks = targets["masks"]  # [Batch, 26, 384, 384]

        # A. Tính Classification Loss
        loss_cls = self.cls_criterion(logits, gt_labels)

        # B. Tính Contrastive Loss
        # (Nếu concept_vectors là None thì bỏ qua - phòng trường hợp debug)
        if concept_vectors is not None:
            loss_con = self.con_criterion(concept_vectors, gt_labels)
        else:
            loss_con = torch.tensor(0.0, device=logits.device)

        # C. Tính Segmentation Loss (Guided by Bbox)
        # Upsample map từ 24x24 lên 384x384 để so khớp với mask bbox
        pred_maps_up = F.interpolate(
            attn_maps, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        loss_seg = self.seg_criterion(pred_maps_up, gt_masks)

        # --- TỔNG HỢP LOSS ---
        total_loss = (
            loss_cls
            + (self.contrastive_weight * loss_con)
            + (self.seg_weight * loss_seg)
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_con": loss_con,
            "loss_seg": loss_seg,
        }
