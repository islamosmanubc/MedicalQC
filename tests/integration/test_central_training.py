from torch.utils.data import DataLoader, Subset

from src.data.datasets import ToyStudyDataset, collate_mil
from src.models.qc_model import QCFederatedMILModel, QCModelConfig
from src.models.spectral import SpectralConfig
from src.train.central_trainer import CentralConfig, train_central


def test_central_training_end_to_end(tmp_path):
    dataset = ToyStudyDataset(
        num_hospitals=3,
        studies_per_hospital=4,
        slices_per_study=4,
        image_size=32,
        seed=0,
    )

    # train on hospital_000 and hospital_001
    train_idxs = [
        i for i, r in enumerate(dataset.records) if r["hospital_id"] != "hospital_002"
    ]
    holdout_idxs = [
        i for i, r in enumerate(dataset.records) if r["hospital_id"] == "hospital_002"
    ]

    train_loader = DataLoader(
        Subset(dataset, train_idxs), batch_size=2, shuffle=True, collate_fn=collate_mil
    )
    holdout_loader = DataLoader(
        Subset(dataset, holdout_idxs),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_mil,
    )

    cfg = QCModelConfig(
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        in_channels=1,
        embed_dim=16,
        lora=None,
        spectral=SpectralConfig(out_dim=8, mode="radial", target_size=32),
        fusion_mode="concat_mlp",
        fusion_dim=16,
        attn_hidden=16,
        dropout=0.0,
        uncertainty_mode="none",
    )
    model = QCFederatedMILModel(cfg)

    history = train_central(
        model,
        train_loader,
        {"holdout": holdout_loader},
        CentralConfig(
            epochs=1,
            device="cpu",
            eval_cfg=CentralConfig().eval_cfg,
            output_dir=str(tmp_path),
        ),
    )

    assert len(history) == 1
