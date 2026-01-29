import torch
from torch import nn

from src.models.spectral import Fusion, SpectralConfig, SpectralEncoder


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_proj = nn.Linear(4, 8)
        self.spec = SpectralEncoder(
            SpectralConfig(out_dim=8, mode="radial", target_size=16)
        )
        self.fusion = Fusion(8, 8, 8, mode="gated", dropout=0.0)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W]
        img = x.mean(dim=(1, 3, 4))  # [B, C]
        img = self.img_proj(img)
        spec = self.spec(x)
        fused = self.fusion(img, spec)
        return self.head(fused).squeeze(-1)


def test_gradients_flow_through_fusion_and_head():
    model = TinyModel()
    x = torch.rand(2, 3, 1, 16, 16)
    out = model(x).mean()
    out.backward()

    assert model.fusion is not None
    assert any(p.grad is not None for p in model.fusion.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
