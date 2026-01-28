"""Spectral encoder for per-slice FFT features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpectralConfig:
    out_dim: int = 128
    mode: str = "radial"  # radial | fft2d
    target_size: int = 224
    radial_bins: Optional[int] = None
    fft_grid: int = 16
    log_scale: bool = True


class SpectralEncoder(nn.Module):
    """Compute spectral embedding from slices using FFT magnitude."""

    def __init__(self, cfg: SpectralConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        if cfg.target_size <= 0:
            raise ValueError("target_size must be > 0")
        if cfg.mode not in {"radial", "fft2d"}:
            raise ValueError(f"Unknown spectral mode: {cfg.mode}")

        if cfg.mode == "radial":
            bins = cfg.radial_bins or cfg.out_dim
            if bins <= 0:
                raise ValueError("radial_bins must be > 0")
            self.radial_bins = bins
            self.register_buffer("_radial_index", _build_radial_index(cfg.target_size, bins))
            self.proj = nn.Identity() if bins == cfg.out_dim else nn.Linear(bins, cfg.out_dim)
        else:
            self.radial_bins = None
            self.register_buffer("_radial_index", torch.zeros(1, dtype=torch.long))
            grid = cfg.fft_grid
            if grid <= 0:
                raise ValueError("fft_grid must be > 0")
            self.fft_grid = grid
            self.proj = nn.Linear(grid * grid, cfg.out_dim)

        self.norm = nn.LayerNorm(cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W] or [B, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError("Expected 4D or 5D tensor")
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)

        x = _center_pad_or_crop(x, self.cfg.target_size)
        x = x.mean(dim=1, keepdim=True)

        fft = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fft)
        if self.cfg.log_scale:
            mag = torch.log1p(mag)

        if self.cfg.mode == "radial":
            features = _radial_pool(mag, self._radial_index, self.radial_bins or self.cfg.out_dim)
        else:
            features = _fft2d_pool(mag, self.fft_grid)

        emb = self.proj(features)
        emb = self.norm(emb)
        emb = emb.reshape(b, s, -1).mean(dim=1)
        return emb


class Fusion(nn.Module):
    """Fuse image and spectral embeddings with configurable modes."""

    def __init__(
        self,
        img_dim: int,
        spec_dim: int,
        out_dim: int,
        mode: str = "concat_mlp",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if img_dim <= 0 or spec_dim <= 0 or out_dim <= 0:
            raise ValueError("dims must be > 0")
        if mode not in {"concat_mlp", "gated", "add"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.mode = mode
        self.out_dim = out_dim

        if mode == "concat_mlp":
            self.net = nn.Sequential(
                nn.Linear(img_dim + spec_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
        elif mode == "gated":
            self.img_proj = nn.Linear(img_dim, out_dim)
            self.spec_proj = nn.Linear(spec_dim, out_dim)
            self.gate = nn.Linear(img_dim + spec_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
        else:
            self.img_proj = nn.Identity() if img_dim == out_dim else nn.Linear(img_dim, out_dim)
            self.spec_proj = nn.Linear(spec_dim, out_dim)

    def forward(self, e_img: torch.Tensor, e_spec: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat_mlp":
            return self.net(torch.cat([e_img, e_spec], dim=1))
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([e_img, e_spec], dim=1)))
            img = self.img_proj(e_img)
            spec = self.spec_proj(e_spec)
            return self.dropout(gate * img + (1 - gate) * spec)
        img = self.img_proj(e_img)
        spec = self.spec_proj(e_spec)
        return img + spec


def _center_pad_or_crop(x: torch.Tensor, target: int) -> torch.Tensor:
    # x: [N, C, H, W]
    _, _, h, w = x.shape
    if h > target:
        top = (h - target) // 2
        x = x[:, :, top : top + target, :]
    elif h < target:
        pad_top = (target - h) // 2
        pad_bottom = target - h - pad_top
        x = F.pad(x, (0, 0, pad_top, pad_bottom))

    _, _, h, w = x.shape
    if w > target:
        left = (w - target) // 2
        x = x[:, :, :, left : left + target]
    elif w < target:
        pad_left = (target - w) // 2
        pad_right = target - w - pad_left
        x = F.pad(x, (pad_left, pad_right, 0, 0))

    return x


def _build_radial_index(size: int, bins: int) -> torch.Tensor:
    fy = torch.fft.fftfreq(size)
    fx = torch.fft.rfftfreq(size)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_y**2 + grid_x**2)
    radius = radius / radius.max().clamp_min(1e-6)
    idx = torch.floor(radius * (bins - 1)).long()
    return idx


def _radial_pool(mag: torch.Tensor, radial_idx: torch.Tensor, bins: int) -> torch.Tensor:
    # mag: [N, 1, H, W2]
    n = mag.shape[0]
    flat = mag.squeeze(1).reshape(n, -1)
    idx = radial_idx.reshape(1, -1).expand(n, -1)
    sums = torch.zeros((n, bins), device=mag.device, dtype=mag.dtype)
    counts = torch.zeros((n, bins), device=mag.device, dtype=mag.dtype)
    sums.scatter_add_(1, idx, flat)
    counts.scatter_add_(1, idx, torch.ones_like(flat))
    return sums / counts.clamp_min(1.0)


def _fft2d_pool(mag: torch.Tensor, grid: int) -> torch.Tensor:
    # mag: [N, 1, H, W2]
    mag = F.interpolate(mag, size=(grid, grid), mode="bilinear", align_corners=False)
    return mag.flatten(1)
