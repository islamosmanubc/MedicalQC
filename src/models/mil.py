"""MIL aggregation modules for study-level decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


class AttentionMILPool(nn.Module):
    """Attention-based MIL pooling over slice embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be > 0")
        if dropout < 0:
            raise ValueError("dropout must be >= 0")
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings: [B, S, D]
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be [B, S, D]")
        weights = self.attn(embeddings)  # [B, S, 1]
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError("mask must be [B, S]")
            weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(weights, dim=1)  # [B, S, 1]
        pooled = (attn * embeddings).sum(dim=1)
        return pooled, attn.squeeze(-1)


class WorstSliceTopK(nn.Module):
    """Top-K mean of per-slice logits (after masking)."""

    def __init__(self, in_dim: int, k: int = 3) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.scorer = nn.Linear(in_dim, 1)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # embeddings: [B, S, D]
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be [B, S, D]")
        logits = self.scorer(embeddings).squeeze(-1)  # [B, S]
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError("mask must be [B, S]")
            logits = logits.masked_fill(~mask, float("-inf"))
        # handle fewer-than-k valid slices by using min(valid, k)
        if mask is None:
            k = min(self.k, logits.size(1))
            topk, _ = torch.topk(logits, k=k, dim=1)
            return topk.mean(dim=1)
        valid_counts = mask.sum(dim=1).clamp_min(1)
        max_k = min(self.k, logits.size(1))
        topk, _ = torch.topk(logits, k=max_k, dim=1)
        # If valid < k, padded logits are -inf, so ignore them by masking
        valid_k = valid_counts.clamp_max(max_k).unsqueeze(1)
        indices = torch.arange(max_k, device=logits.device).unsqueeze(0)
        keep = indices < valid_k
        topk = topk.masked_fill(~keep, 0.0)
        denom = valid_k.squeeze(1).float()
        return topk.sum(dim=1) / denom


@dataclass(frozen=True)
class HybridAggregatorConfig:
    in_dim: int
    attn_hidden: int = 256
    topk: int = 3
    fusion_mode: str = "embed"  # embed | logit
    alpha: float = 0.5
    out_dim: int = 256
    dropout: float = 0.1


class HybridStudyAggregator(nn.Module):
    """Hybrid of attention pooling and worst-slice TopK aggregation."""

    def __init__(self, cfg: HybridAggregatorConfig) -> None:
        super().__init__()
        if cfg.alpha < 0.0 or cfg.alpha > 1.0:
            raise ValueError("alpha must be in [0,1]")
        if cfg.fusion_mode not in {"embed", "logit"}:
            raise ValueError("fusion_mode must be 'embed' or 'logit'")
        self.cfg = cfg
        self.attn_pool = AttentionMILPool(cfg.in_dim, cfg.attn_hidden, cfg.dropout)
        self.topk_pool = WorstSliceTopK(cfg.in_dim, cfg.topk)
        if cfg.fusion_mode == "embed":
            self.topk_proj = nn.Linear(1, cfg.out_dim)
            self.attn_proj = nn.Linear(cfg.in_dim, cfg.out_dim)
        else:
            self.logit_head = nn.Linear(cfg.in_dim, 1)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_emb, attn_weights = self.attn_pool(embeddings, mask)
        topk_logits = self.topk_pool(embeddings, mask).unsqueeze(-1)

        if self.cfg.fusion_mode == "embed":
            emb = self.attn_proj(attn_emb)
            topk_emb = self.topk_proj(topk_logits)
            fused = self.cfg.alpha * emb + (1 - self.cfg.alpha) * topk_emb
            return fused, {"attn_weights": attn_weights, "topk_logits": topk_logits.squeeze(-1)}

        # logit-level fusion
        attn_logit = self.logit_head(attn_emb)
        fused_logit = self.cfg.alpha * attn_logit + (1 - self.cfg.alpha) * topk_logits
        return fused_logit.squeeze(-1), {"attn_weights": attn_weights, "topk_logits": topk_logits.squeeze(-1)}
