"""DataLoader builders for MIL and federated splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.datasets import (
    PreprocessConfig,
    SamplingConfig,
    StudyFolderDataset,
    ToyStudyDataset,
    collate_mil,
)
from src.data.samplers import FederatedClientSampler


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    seed: int = 0


@dataclass(frozen=True)
class DataConfig:
    root: str | None = None
    backend: str = "folder"  # folder | toy
    preprocess: PreprocessConfig = PreprocessConfig()
    sampling: SamplingConfig = SamplingConfig()
    seed: int = 0
    # toy dataset options
    num_hospitals: int = 3
    studies_per_hospital: int = 4
    slices_per_study: int = 8
    image_size: int = 64


def build_dataset(cfg: DataConfig) -> StudyFolderDataset | ToyStudyDataset:
    if cfg.backend == "folder":
        if cfg.root is None:
            raise ValueError("DataConfig.root is required for folder backend")
        return StudyFolderDataset(
            root=Path(cfg.root),
            preprocess=cfg.preprocess,
            sampling=cfg.sampling,
            seed=cfg.seed,
        )
    if cfg.backend == "toy":
        return ToyStudyDataset(
            num_hospitals=cfg.num_hospitals,
            studies_per_hospital=cfg.studies_per_hospital,
            slices_per_study=cfg.slices_per_study,
            image_size=cfg.image_size,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown backend: {cfg.backend}")


def build_dataloader_global(cfg: DataConfig, loader_cfg: LoaderConfig) -> DataLoader:
    dataset = build_dataset(cfg)
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last,
        collate_fn=collate_mil,
    )


def build_dataloader_for_hospital(
    cfg: DataConfig, loader_cfg: LoaderConfig, hospital_id: str
) -> DataLoader:
    dataset = build_dataset(cfg)
    if not hasattr(dataset, "hospital_to_indices"):
        raise ValueError("Dataset does not support hospital splits")
    indices = dataset.hospital_to_indices.get(hospital_id)
    if not indices:
        raise ValueError(f"No samples found for hospital_id={hospital_id}")
    sampler = FederatedClientSampler(indices, shuffle=True, seed=loader_cfg.seed)
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        sampler=sampler,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last,
        collate_fn=collate_mil,
    )
