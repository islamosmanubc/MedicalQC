"""Hydra configuration schemas for canonical src package."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.lora import LoraConfig
from src.models.qc_model import QCModelConfig
from src.models.spectral import SpectralConfig


@dataclass
class DataConfig:
    dataset_root: str = "${oc.env:DATA_ROOT,./data}"
    dataset_type: str = "toy"  # toy | folder
    num_hospitals: int = 3
    studies_per_hospital: int = 4
    slices_per_study: int = 8
    image_size: int = 64
    seed: int = 7
    num_workers: int = 0
    pin_memory: bool = True
    sampling_max_slices: int | None = None
    sampling_strategy: str = "uniform"
    ct_window_level: float | None = None
    ct_window_width: float | None = None
    mri_norm: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 3
    device: str = "cpu"
    mixed_precision: bool = False
    grad_clip_norm: float = 1.0
    seed: int = 42
    deterministic: bool = True


@dataclass
class FederatedConfig:
    num_clients: int = 3
    rounds: int = 3
    local_epochs: int = 1
    holdout_clients: list[int] = field(default_factory=lambda: [2])


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class MlflowConfig:
    tracking_uri: str = "${oc.env:MLFLOW_TRACKING_URI,./mlruns}"
    experiment_name: str = "medicalqc"
    run_name: str = "${now:%Y-%m-%d_%H-%M-%S}"


@dataclass
class EvalConfig:
    specificity_target: float = 0.95
    sensitivity_target: float = 0.8


@dataclass
class ModelConfig:
    encoder_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = False
    in_channels: int = 1
    embed_dim: int = 64
    lora: LoraConfig | None = None
    spectral: SpectralConfig = SpectralConfig()
    fusion_mode: str = "concat_mlp"
    fusion_dim: int = 32
    attn_hidden: int = 32
    dropout: float = 0.1
    uncertainty_mode: str = "none"
    return_ci: bool = False
    expected_modality: str | None = None
    freeze_backbone: bool = False
    train_adapters_only: bool = False

    def to_qc_config(self) -> QCModelConfig:
        return QCModelConfig(
            encoder_name=self.encoder_name,
            pretrained=self.pretrained,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            lora=self.lora,
            spectral=self.spectral,
            fusion_mode=self.fusion_mode,
            fusion_dim=self.fusion_dim,
            attn_hidden=self.attn_hidden,
            dropout=self.dropout,
            uncertainty_mode=self.uncertainty_mode,
            return_ci=self.return_ci,
            expected_modality=self.expected_modality,
            freeze_backbone=self.freeze_backbone,
            train_adapters_only=self.train_adapters_only,
        )


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "${hydra:runtime.output_dir}"
