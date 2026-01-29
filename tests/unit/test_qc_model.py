import torch

from src.models.lora import LoraConfig
from src.models.qc_model import QCFederatedMILModel, QCModelConfig
from src.models.spectral import SpectralConfig


def test_qc_model_forward_with_padding():
    cfg = QCModelConfig(
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        in_channels=1,
        embed_dim=64,
        lora=None,
        spectral=SpectralConfig(out_dim=16, mode="radial", target_size=32),
        fusion_mode="concat_mlp",
        fusion_dim=32,
        attn_hidden=32,
        dropout=0.0,
        uncertainty_mode="none",
    )
    model = QCFederatedMILModel(cfg)
    slices = torch.rand(2, 4, 1, 32, 32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    batch = {"slices": slices, "attention_mask": mask}
    out = model(batch)
    assert out["logits"].shape == (2,)
    assert out["p_fail"].shape == (2,)
    assert out["u"].shape == (2,)
    assert out["attention_weights"].shape == (2, 4)


def test_shared_private_state_roundtrip():
    cfg = QCModelConfig(
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        in_channels=1,
        embed_dim=32,
        lora=LoraConfig(r=2, alpha=4.0, dropout=0.0, target_modules=("qkv", "proj")),
        spectral=SpectralConfig(out_dim=8, mode="fft2d", target_size=32, fft_grid=8),
        fusion_mode="add",
        fusion_dim=16,
        attn_hidden=16,
        dropout=0.0,
        uncertainty_mode="none",
    )
    model = QCFederatedMILModel(cfg)
    shared = model.shared_state_dict()
    private = model.private_state_dict()

    model2 = QCFederatedMILModel(cfg)
    model2.load_shared_state_dict(shared)
    model2.load_private_state_dict(private)

    for key, value in private.items():
        assert torch.allclose(model2.state_dict()[key], value)
