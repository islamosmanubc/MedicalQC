"""LoRA utilities for Linear layers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")

    def validate(self) -> None:
        if self.r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be > 0")
        if self.dropout < 0:
            raise ValueError("LoRA dropout must be >= 0")
        if not self.target_modules:
            raise ValueError("LoRA target_modules must not be empty")


class LoRALinear(nn.Module):
    """LoRA wrapper for linear layers."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros((r, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora = self.dropout(x) @ self.lora_A.t()
        lora = lora @ self.lora_B.t()
        return result + lora * self.scaling


def inject_lora(module: nn.Module, config: LoraConfig) -> nn.Module:
    """Replace target Linear modules with LoRA-enabled versions.

    The LoRA parameters are stored under keys containing "lora_A" and "lora_B".
    """
    config.validate()
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _matches(name, config.target_modules):
            setattr(
                module, name, LoRALinear(child, config.r, config.alpha, config.dropout)
            )
        else:
            inject_lora(child, config)
    return module


def shared_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return only shared weights (exclude LoRA parameters)."""
    return {k: v for k, v in state_dict.items() if not _is_lora_key(k)}


def private_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return only private LoRA weights."""
    return {k: v for k, v in state_dict.items() if _is_lora_key(k)}


def load_private_state_dict(
    model: nn.Module, lora_state: dict[str, torch.Tensor]
) -> None:
    """Load LoRA-only state into model."""
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys: {unexpected}")
    # missing keys are allowed if a layer does not have LoRA


def _is_lora_key(key: str) -> bool:
    return key.endswith("lora_A") or key.endswith("lora_B")


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)
