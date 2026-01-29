"""Federated training wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from src.federated.client import FederatedClient
from src.federated.runner import RunnerConfig, run_federated
from src.federated.server import FederatedServer


@dataclass(frozen=True)
class FederatedTrainerConfig:
    runner: RunnerConfig = RunnerConfig()


def train_federated(
    server: FederatedServer,
    clients: list[FederatedClient],
    holdout_loaders: dict[str, DataLoader],
    cfg: FederatedTrainerConfig,
) -> list[dict[str, float]]:
    return run_federated(server, clients, holdout_loaders, cfg.runner)
