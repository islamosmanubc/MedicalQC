import torch

from src.models.spectral import SpectralConfig, SpectralEncoder


def test_spectral_encoder_finite_zero_and_constant():
    cfg = SpectralConfig(out_dim=8, mode="radial", target_size=32)
    encoder = SpectralEncoder(cfg)

    zeros = torch.zeros(2, 3, 1, 20, 24)
    ones = torch.ones(2, 3, 1, 20, 24)

    out_zero = encoder(zeros)
    out_one = encoder(ones)

    assert torch.isfinite(out_zero).all()
    assert torch.isfinite(out_one).all()


def test_spectral_encoder_cpu_forward():
    cfg = SpectralConfig(out_dim=8, mode="fft2d", target_size=32, fft_grid=8)
    encoder = SpectralEncoder(cfg)
    x = torch.rand(1, 2, 1, 28, 36)
    out = encoder(x)
    assert out.shape == (1, 8)
