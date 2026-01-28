"""Uncertainty heads and calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EvidentialConfig:
    in_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1


class EvidentialBinaryHead(nn.Module):
    """Evidential head for binary classification using Beta distribution."""

    def __init__(self, cfg: EvidentialConfig) -> None:
        super().__init__()
        if cfg.in_dim <= 0 or cfg.hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be > 0")
        if cfg.dropout < 0:
            raise ValueError("dropout must be >= 0")
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(self.net(x))
        alpha = evidence[:, 0] + 1.0
        beta = evidence[:, 1] + 1.0
        total = alpha + beta
        p_fail = alpha / total
        u = 2.0 / total
        return {
            "p_fail": p_fail,
            "u": u,
            "alpha": alpha,
            "beta": beta,
        }


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for calibration."""

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        if init_temp <= 0:
            raise ValueError("init_temp must be > 0")
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimize temperature on validation set. Returns final NLL."""
        if logits.ndim != 1:
            logits = logits.view(-1)
        labels = labels.float().view(-1)
        optimizer = torch.optim.LBFGS([self.log_temp], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
        return float(loss.item())


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    forward_fn,
    passes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MC Dropout to estimate mean probability and uncertainty variance."""
    if passes <= 1:
        raise ValueError("passes must be > 1")
    enable_mc_dropout(model)
    probs = []
    with torch.no_grad():
        for _ in range(passes):
            logits = forward_fn(model, inputs)
            probs.append(torch.sigmoid(logits))
    stack = torch.stack(probs, dim=0)
    p_mean = stack.mean(dim=0)
    var = stack.var(dim=0)
    return p_mean, var


def expected_calibration_error(
    y_true: torch.Tensor, y_prob: torch.Tensor, n_bins: int = 10
) -> torch.Tensor:
    y_true = y_true.float().view(-1)
    y_prob = y_prob.view(-1)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=y_prob.device)
    ece = torch.tensor(0.0, device=y_prob.device)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.float().mean() * torch.abs(acc - conf)
    return ece


def reliability_diagram(
    y_true: torch.Tensor, y_prob: torch.Tensor, n_bins: int = 10
) -> Dict[str, list]:
    y_true = y_true.float().view(-1)
    y_prob = y_prob.view(-1)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=y_prob.device)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_count.append(0)
            continue
        bin_acc.append(float(y_true[mask].mean().item()))
        bin_conf.append(float(y_prob[mask].mean().item()))
        bin_count.append(int(mask.sum().item()))
    return {
        "bin_edges": [float(x) for x in bins.cpu().tolist()],
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
    }


def decision_policy(
    p_fail: torch.Tensor, u: torch.Tensor, t_fail: float, t_u: float
) -> torch.Tensor:
    """Return decision: 0=PASS, 1=FAIL, 2=UNCERTAIN."""
    if t_fail < 0 or t_fail > 1:
        raise ValueError("t_fail must be in [0,1]")
    if t_u < 0:
        raise ValueError("t_u must be >= 0")
    decision = torch.zeros_like(p_fail, dtype=torch.long)
    decision = torch.where(u >= t_u, torch.tensor(2, device=p_fail.device), decision)
    decision = torch.where((u < t_u) & (p_fail >= t_fail), torch.tensor(1, device=p_fail.device), decision)
    return decision
