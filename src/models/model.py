"""Composite model with Swin encoder + spectral branch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.spectral import Fusion, SpectralConfig, SpectralEncoder
from src.models.swin_encoder import SwinEncoder


class MILAttention(nn.Module):
    """Attention pooling over slice embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, S, D]
        weights = self.attn(x)  # [B, S, 1]
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled


class SpectralMILModel(nn.Module):
    """Swin encoder + spectral branch + fusion + head."""

    def __init__(
        self,
        encoder: SwinEncoder,
        spectral_cfg: SpectralConfig,
        fusion_mode: str = "concat_mlp",
        fusion_dim: int = 256,
        mil_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.spectral = SpectralEncoder(spectral_cfg)
        self.mil = MILAttention(encoder.out_dim, mil_hidden, dropout)
        self.fusion = Fusion(
            encoder.out_dim, spectral_cfg.out_dim, fusion_dim, fusion_mode, dropout
        )
        self.head = nn.Linear(fusion_dim, 1)

    def forward(
        self, slices: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if slices.dim() != 5:
            raise ValueError("Expected slices to be [B, S, C, H, W]")
        b, s, c, h, w = slices.shape
        flat = slices.view(b * s, c, h, w)
        feats = self.encoder(flat).view(b, s, -1)
        pooled = self.mil(feats, mask=attention_mask)
        spec = self.spectral(slices)
        fused = self.fusion(pooled, spec)
        logit = self.head(fused).squeeze(-1)
        return {"logit": logit, "e_img": pooled, "e_spec": spec}
