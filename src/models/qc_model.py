"""Federated QC model with MIL, spectral fusion, and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from src.models.lora import LoraConfig, load_private_state_dict, private_state_dict, shared_state_dict
from src.models.mil import AttentionMILPool
from src.models.spectral import Fusion, SpectralConfig, SpectralEncoder
from src.models.swin_encoder import SwinEncoder
from src.models.uncertainty import EvidentialBinaryHead, EvidentialConfig


@dataclass(frozen=True)
class QCModelConfig:
    encoder_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = False
    in_channels: int = 1
    embed_dim: int = 768
    lora: Optional[LoraConfig] = None
    spectral: SpectralConfig = SpectralConfig()
    fusion_mode: str = "concat_mlp"
    fusion_dim: int = 256
    attn_hidden: int = 256
    dropout: float = 0.1
    uncertainty_mode: str = "none"  # none | evidential
    return_ci: bool = False
    expected_modality: Optional[str] = None  # CT | MRI | None
    freeze_backbone: bool = False
    train_adapters_only: bool = False


class QCFederatedMILModel(nn.Module):
    """Swin encoder + spectral branch + MIL attention + uncertainty head."""

    def __init__(self, cfg: QCModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if cfg.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if cfg.dropout < 0:
            raise ValueError("dropout must be >= 0")
        if cfg.uncertainty_mode not in {"none", "evidential"}:
            raise ValueError("uncertainty_mode must be 'none' or 'evidential'")

        self.encoder = SwinEncoder(
            name=cfg.encoder_name,
            pretrained=cfg.pretrained,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            lora=cfg.lora,
        )
        self.spectral = SpectralEncoder(cfg.spectral)
        self.mil = AttentionMILPool(cfg.embed_dim, cfg.attn_hidden, cfg.dropout)
        self.fusion = Fusion(cfg.embed_dim, cfg.spectral.out_dim, cfg.fusion_dim, cfg.fusion_mode, cfg.dropout)
        self.head = nn.Linear(cfg.fusion_dim, 1)

        if cfg.uncertainty_mode == "evidential":
            self.uncertainty_head = EvidentialBinaryHead(
                EvidentialConfig(in_dim=cfg.fusion_dim, hidden_dim=cfg.attn_hidden, dropout=cfg.dropout)
            )
        else:
            self.uncertainty_head = None

        if cfg.freeze_backbone:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = False

        if cfg.train_adapters_only:
            self._freeze_all_except_lora()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        slices = batch["slices"]
        attention_mask = batch.get("attention_mask")
        if slices.dim() != 5:
            raise ValueError("slices must be [B, S, C, H, W]")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError("attention_mask must be [B, S]")

        if self.cfg.expected_modality is not None and "meta" in batch:
            self._validate_modality(batch["meta"], self.cfg.expected_modality)

        b, s, c, h, w = slices.shape
        flat = slices.view(b * s, c, h, w)
        slice_embeddings = self.encoder(flat).view(b, s, -1)
        pooled, attn_weights = self.mil(slice_embeddings, attention_mask)
        spec = self.spectral(slices)
        fused = self.fusion(pooled, spec)
        logits = self.head(fused).squeeze(-1)

        if self.uncertainty_head is not None:
            unc = self.uncertainty_head(fused)
            p_fail = unc["p_fail"]
            u = unc["u"]
            output = {
                "logits": logits,
                "p_fail": p_fail,
                "u": u,
                "attention_weights": attn_weights,
            }
            if self.cfg.return_ci:
                alpha = unc["alpha"]
                beta = unc["beta"]
                mean = p_fail
                var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                ci = 1.96 * torch.sqrt(var)
                output["ci_low"] = torch.clamp(mean - ci, 0.0, 1.0)
                output["ci_high"] = torch.clamp(mean + ci, 0.0, 1.0)
            return output

        p_fail = torch.sigmoid(logits)
        u = torch.zeros_like(p_fail)
        return {
            "logits": logits,
            "p_fail": p_fail,
            "u": u,
            "attention_weights": attn_weights,
        }

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        return shared_state_dict(self.state_dict())

    def private_state_dict(self) -> Dict[str, torch.Tensor]:
        return private_state_dict(self.state_dict())

    def load_shared_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for key in state:
            if key.endswith("lora_A") or key.endswith("lora_B"):
                raise ValueError("Shared state dict contains LoRA parameters")
        self.load_state_dict(state, strict=False)

    def load_private_state_dict(self, lora_state: Dict[str, torch.Tensor]) -> None:
        load_private_state_dict(self, lora_state)

    def _freeze_all_except_lora(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")

    @staticmethod
    def _validate_modality(meta, expected: str) -> None:
        expected = expected.upper()
        if isinstance(meta, list):
            modalities = {str(m.get("modality", "")).upper() for m in meta}
            if expected not in modalities:
                raise ValueError(f"Expected modality {expected}, got {modalities}")
        elif isinstance(meta, dict):
            modality = str(meta.get("modality", "")).upper()
            if modality != expected:
                raise ValueError(f"Expected modality {expected}, got {modality}")
