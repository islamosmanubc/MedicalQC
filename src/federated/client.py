"""Federated client with private LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


@dataclass
class ClientConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    local_epochs: int = 1
    mixed_precision: bool = False
    grad_clip_norm: float = 1.0


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: ClientConfig,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.scaler = GradScaler(enabled=cfg.mixed_precision)

    def get_shared_state(self) -> dict[str, torch.Tensor]:
        if hasattr(self.model, "shared_state_dict"):
            return {
                k: v.detach().cpu() for k, v in self.model.shared_state_dict().items()
            }
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def get_private_state(self) -> dict[str, torch.Tensor]:
        if hasattr(self.model, "private_state_dict"):
            return {
                k: v.detach().cpu() for k, v in self.model.private_state_dict().items()
            }
        return {}

    def load_shared_state(self, state: dict[str, torch.Tensor]) -> None:
        if hasattr(self.model, "load_shared_state_dict"):
            self.model.load_shared_state_dict(state)
        else:
            self.model.load_state_dict(state, strict=False)

    def load_private_state(self, state: dict[str, torch.Tensor]) -> None:
        if hasattr(self.model, "load_private_state_dict"):
            self.model.load_private_state_dict(state)

    def train(self) -> dict[str, float]:
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        steps = 0

        for _ in range(self.cfg.local_epochs):
            for batch in self.loader:
                optimizer.zero_grad(set_to_none=True)
                batch = _move_batch(batch, self.device)
                with autocast(enabled=self.cfg.mixed_precision):
                    outputs = self.model(batch)
                    logits = outputs["logits"]
                    labels = batch["label"]
                    loss = loss_fn(logits, labels)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += float(loss.detach().cpu())
                steps += 1
        return {"train_loss": total_loss / max(steps, 1)}


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
