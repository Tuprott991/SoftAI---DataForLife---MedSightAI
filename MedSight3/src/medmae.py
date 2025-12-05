import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class MedMAEBackbone(nn.Module):
    def __init__(self, model_name='facebook/vit-mae-base'):
        super(MedMAEBackbone, self).__init__()
        # Load pre-trained ViT/MAE
        # Lưu ý: MedMAE thực chất là ViT được train theo kiểu MAE
        # Bạn cần thay 'model_name' bằng đường dẫn tới weights MedMAE thực tế của bạn
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Lấy hidden dimension (ví dụ: ViT-Base là 768)
        self.embed_dim = self.vit.config.hidden_size
        
        # Patch size (thường là 16)
        self.patch_size = self.vit.config.patch_size
        
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W). Ví dụ: 224x224
        Returns:
            feature_map: (B, Embed_Dim, Grid_H, Grid_W). Ví dụ: (B, 768, 14, 14)
        """
        # 1. Forward qua ViT
        # outputs.last_hidden_state shape: (B, Sequence_Length, Hidden_Dim)
        # Sequence_Length = 1 (CLS token) + (H*W)/(P*P) patches
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Loại bỏ CLS token (token đầu tiên dùng để phân loại chung)
        # Chúng ta cần features của từng vùng ảnh cho CSR
        patch_tokens = last_hidden_state[:, 1:, :] # (B, 196, 768) với ảnh 224, patch 16
        
        # 3. Reshape từ Sequence về 2D Grid
        # Tính kích thước lưới grid: H_grid = H_img // patch_size
        B, N, C = patch_tokens.shape
        H_grid = int(N**0.5) # Căn bậc 2 của số patch (ví dụ: căn(196) = 14)
        W_grid = H_grid
        
        # Permute để đưa Channel lên trước: (B, H*W, C) -> (B, C, H*W)
        patch_tokens = patch_tokens.permute(0, 2, 1)
        
        # View lại thành 2D: (B, C, H_grid, W_grid)
        feature_map = patch_tokens.view(B, C, H_grid, W_grid)
        
        return feature_map

    @property
    def out_channels(self):
        # Property để CSR biết dimension đầu ra (thay vì fc.in_features của ResNet)
        return self.embed_dim