"""Swin-T encoder wrapper with optional LoRA adapters."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

import timm

from src.models.lora import (
    LoraConfig,
    inject_lora,
    load_private_state_dict,
    private_state_dict,
    shared_state_dict,
)


class SwinEncoder(nn.Module):
    """Swin-T backbone returning per-slice embeddings."""

    def __init__(
        self,
        name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = False,
        in_channels: int = 1,
        embed_dim: Optional[int] = None,
        lora: Optional[LoraConfig] = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self._adapt_input(in_channels)
        self.backbone_dim = int(getattr(self.backbone, "num_features", 768))
        out_dim = embed_dim or self.backbone_dim
        if out_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        self.proj = nn.Identity() if out_dim == self.backbone_dim else nn.Linear(self.backbone_dim, out_dim)
        self.out_dim = out_dim
        self.lora_config = lora
        if lora is not None:
            inject_lora(self.backbone, lora)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        return self.proj(features)

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        return shared_state_dict(self.state_dict())

    def private_state_dict(self) -> Dict[str, torch.Tensor]:
        return private_state_dict(self.state_dict())

    def load_private_state_dict(self, lora_state: Dict[str, torch.Tensor]) -> None:
        load_private_state_dict(self, lora_state)

    def load_shared_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for key in state:
            if key.endswith("lora_A") or key.endswith("lora_B"):
                raise ValueError("Shared state dict contains LoRA parameters")
        self.load_state_dict(state, strict=False)

    def _adapt_input(self, in_channels: int) -> None:
        if in_channels == 3:
            return
        if not hasattr(self.backbone, "patch_embed"):
            return
        proj = self.backbone.patch_embed.proj
        if proj.in_channels == in_channels:
            return
        new_proj = nn.Conv2d(
            in_channels,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_proj.weight.copy_(proj.weight.mean(dim=1, keepdim=True))
            else:
                new_proj.weight[:, : proj.in_channels].copy_(proj.weight)
        self.backbone.patch_embed.proj = new_proj
