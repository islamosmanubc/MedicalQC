import torch
from torch import nn

from src.models.lora import LoraConfig, inject_lora


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(x))


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_lora_injection_increases_params():
    model = TinyBlock()
    before = count_trainable_params(model)
    inject_lora(
        model, LoraConfig(r=2, alpha=4.0, dropout=0.0, target_modules=("qkv", "proj"))
    )
    after = count_trainable_params(model)
    assert after > before
