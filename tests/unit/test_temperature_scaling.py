import torch

from src.models.uncertainty import TemperatureScaler


def test_temperature_scaling_improves_nll():
    torch.manual_seed(0)
    # Overconfident logits for a noisy binary label set
    logits = torch.tensor([6.0, -6.0, 6.0, -6.0, 6.0, -6.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    nll_before = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels
    ).item()
    scaler = TemperatureScaler(init_temp=1.0)
    nll_after = scaler.fit(logits, labels, max_iter=50)

    assert nll_after <= nll_before
