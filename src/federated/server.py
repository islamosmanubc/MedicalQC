"""Federated server for shared model aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn

from src.federated.aggregation import Aggregator, FedAvg


class SecureAggregation(ABC):
    """Placeholder secure aggregation interface."""

    @abstractmethod
    def aggregate(
        self,
        aggregator: Aggregator,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class NoSecureAggregation(SecureAggregation):
    def aggregate(
        self,
        aggregator: Aggregator,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        return aggregator.aggregate(global_state, client_states, weights)


@dataclass
class FederatedServer:
    model: nn.Module
    aggregator: Aggregator = field(default_factory=FedAvg)
    secure_agg: SecureAggregation = field(default_factory=NoSecureAggregation)

    def get_shared_state(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self.model, "shared_state_dict"):
            return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return {k: v.detach().cpu() for k, v in self.model.shared_state_dict().items()}

    def load_shared_state(self, state: Dict[str, torch.Tensor]) -> None:
        if hasattr(self.model, "load_shared_state_dict"):
            self.model.load_shared_state_dict(state)
        else:
            self.model.load_state_dict(state, strict=False)

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> None:
        global_state = self.get_shared_state()
        new_state = self.secure_agg.aggregate(self.aggregator, global_state, client_states, weights)
        self.load_shared_state(new_state)
