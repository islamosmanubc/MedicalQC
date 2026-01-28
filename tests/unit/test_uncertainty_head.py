import torch

from src.models.uncertainty import EvidentialBinaryHead, EvidentialConfig


def test_evidential_outputs_in_range():
    head = EvidentialBinaryHead(EvidentialConfig(in_dim=4, hidden_dim=8, dropout=0.0))
    x = torch.randn(3, 4)
    out = head(x)
    assert (out["p_fail"] >= 0).all() and (out["p_fail"] <= 1).all()
    assert (out["u"] >= 0).all() and (out["u"] <= 1).all()
