import torch
from src.model import CSR

# Test actual dimensions
model = CSR(22, 6, 1, 'medmae', 'weights/chestx-medmae_finetune.pth')
x = torch.randn(1, 3, 224, 224)
out = model(x)

print("Actual feature map dimensions:")
print(f"Features: {out['features'].shape}")
print(f"CAMs: {out['cams'].shape}")
print(f"Similarity maps: {out['similarity_maps'].shape}")
