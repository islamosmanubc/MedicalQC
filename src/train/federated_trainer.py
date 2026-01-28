"""Federated training wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from torch.utils.data import DataLoader

from src.federated.runner import RunnerConfig, run_federated
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient


@dataclass(frozen=True)
class FederatedTrainerConfig:
    runner: RunnerConfig = RunnerConfig()


def train_federated(
    server: FederatedServer,
    clients: List[FederatedClient],
    holdout_loaders: Dict[str, DataLoader],
    cfg: FederatedTrainerConfig,
) -> List[Dict[str, float]]:
    return run_federated(server, clients, holdout_loaders, cfg.runner)
