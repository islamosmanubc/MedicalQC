import torch
from torch import nn

from src.models.lora import (
    LoraConfig,
    inject_lora,
    load_private_state_dict,
    private_state_dict,
    shared_state_dict,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(x))


def test_lora_state_split_and_reload():
    model = TinyBlock()
    inject_lora(
        model, LoraConfig(r=2, alpha=4.0, dropout=0.0, target_modules=("qkv", "proj"))
    )

    # mutate LoRA params
    for name, param in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            torch.nn.init.constant_(param, 0.5)

    shared = shared_state_dict(model.state_dict())
    private = private_state_dict(model.state_dict())

    assert shared
    assert private
    assert all(not k.endswith("lora_A") and not k.endswith("lora_B") for k in shared)
    assert all(k.endswith("lora_A") or k.endswith("lora_B") for k in private)

    model2 = TinyBlock()
    inject_lora(
        model2, LoraConfig(r=2, alpha=4.0, dropout=0.0, target_modules=("qkv", "proj"))
    )
    load_private_state_dict(model2, private)

    for key, value in private.items():
        assert torch.allclose(model2.state_dict()[key], value)
