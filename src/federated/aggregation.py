"""Federated aggregation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        global_state: dict[str, torch.Tensor],
        client_states: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class FedAvg(Aggregator):
    def aggregate(
        self,
        global_state: dict[str, torch.Tensor],
        client_states: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states provided")
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")
        new_state: dict[str, torch.Tensor] = {}
        for key in global_state:
            acc = None
            for state, w in zip(client_states, weights, strict=False):
                val = state[key].detach().float() * (w / total)
                acc = val if acc is None else acc + val
            new_state[key] = acc.to(global_state[key].dtype)
        return new_state


class FedAdam(Aggregator):
    """FedAdam server optimizer (FedOpt)."""

    def __init__(
        self,
        lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        global_state: dict[str, torch.Tensor],
        client_states: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states provided")
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")

        new_state: dict[str, torch.Tensor] = {}
        for key in global_state:
            delta = None
            for state, w in zip(client_states, weights, strict=False):
                diff = (
                    state[key].detach().float() - global_state[key].detach().float()
                ) * (w / total)
                delta = diff if delta is None else delta + diff
            if key not in self.m:
                self.m[key] = torch.zeros_like(delta)
                self.v[key] = torch.zeros_like(delta)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (delta * delta)
            update = self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
            new_state[key] = (global_state[key].detach().float() + self.lr * update).to(
                global_state[key].dtype
            )
        return new_state
