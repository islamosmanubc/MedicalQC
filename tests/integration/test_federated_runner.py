import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import ToyStudyDataset, collate_mil
from src.federated import (
    FedAvg,
    FederatedClient,
    FederatedServer,
    RunnerConfig,
    run_federated,
)
from src.federated.client import ClientConfig
from src.models.lora import LoraConfig
from src.models.qc_model import QCFederatedMILModel, QCModelConfig
from src.models.spectral import SpectralConfig


def test_federated_runner_loss_decreases(tmp_path):
    torch.manual_seed(0)
    dataset = ToyStudyDataset(
        num_hospitals=3,
        studies_per_hospital=4,
        slices_per_study=4,
        image_size=32,
        seed=0,
    )

    hospital_indices = {}
    for idx, rec in enumerate(dataset.records):
        hospital_indices.setdefault(rec["hospital_id"], []).append(idx)

    loaders = {}
    for hid, idxs in hospital_indices.items():
        subset = Subset(dataset, idxs)
        loaders[hid] = DataLoader(
            subset, batch_size=2, shuffle=True, collate_fn=collate_mil
        )

    cfg = QCModelConfig(
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        in_channels=1,
        embed_dim=16,
        lora=LoraConfig(r=2, alpha=4.0, dropout=0.0, target_modules=("qkv", "proj")),
        spectral=SpectralConfig(out_dim=8, mode="radial", target_size=32),
        fusion_mode="concat_mlp",
        fusion_dim=16,
        attn_hidden=16,
        dropout=0.0,
        uncertainty_mode="none",
    )

    server_model = QCFederatedMILModel(cfg)
    server = FederatedServer(server_model, aggregator=FedAvg())

    clients = []
    holdout_loaders = {"holdout": loaders["hospital_002"]}
    for hid in ["hospital_000", "hospital_001"]:
        client_model = QCFederatedMILModel(cfg)
        client_model.load_shared_state_dict(server.get_shared_state())
        client = FederatedClient(
            client_id=hid,
            model=client_model,
            loader=loaders[hid],
            device=torch.device("cpu"),
            cfg=ClientConfig(lr=1e-2, local_epochs=1, mixed_precision=False),
        )
        clients.append(client)

    history = run_federated(
        server=server,
        clients=clients,
        holdout_loaders=holdout_loaders,
        cfg=RunnerConfig(rounds=2, output_dir=str(tmp_path), log_client_metrics=False),
    )

    assert len(history) == 2
    assert history[-1]["train_loss"] <= history[0]["train_loss"]
