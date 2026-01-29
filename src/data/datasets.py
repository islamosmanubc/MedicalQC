"""Data handling for MIL and federated splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PreprocessConfig:
    ct_window_level: float | None = None
    ct_window_width: float | None = None
    mri_norm: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    max_slices: int | None = None
    strategy: str = "uniform"  # uniform | center | random


@dataclass
class StudyRecord:
    hospital_id: str
    study_id: str
    study_path: Path
    meta: dict[str, Any]


class StudyFolderDataset(Dataset[dict[str, Any]]):
    """Study folder dataset: root/hospital_x/study_y/*"""

    def __init__(
        self,
        root: str | Path,
        hospitals: Iterable[str] | None = None,
        preprocess: PreprocessConfig | None = None,
        sampling: SamplingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.hospitals = sorted(hospitals) if hospitals is not None else None
        self.preprocess = preprocess or PreprocessConfig()
        self.sampling = sampling or SamplingConfig()
        self.seed = seed
        self.records: list[StudyRecord] = self._index_studies()
        self.hospital_to_indices = self._build_hospital_index()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        slice_paths = _list_slices(record.study_path)
        slice_paths = _sample_slices(
            slice_paths,
            record.study_id,
            self.sampling.max_slices,
            self.sampling.strategy,
            self.seed,
        )
        slices = [_load_slice(p) for p in slice_paths]
        slices = [_apply_preprocess(x, record.meta, self.preprocess) for x in slices]
        tensor = torch.stack([_to_tensor(x) for x in slices], dim=0)
        label = float(record.meta.get("label", 0))
        return {
            "hospital_id": record.hospital_id,
            "study_id": record.study_id,
            "slices": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "meta": record.meta,
        }

    def _index_studies(self) -> list[StudyRecord]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        hospitals = [p for p in self.root.iterdir() if p.is_dir()]
        if self.hospitals is not None:
            hospitals = [p for p in hospitals if p.name in set(self.hospitals)]
        records: list[StudyRecord] = []
        for hospital in sorted(hospitals, key=lambda p: p.name):
            for study in sorted(hospital.iterdir(), key=lambda p: p.name):
                if not study.is_dir():
                    continue
                meta_path = study / "meta.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                records.append(
                    StudyRecord(
                        hospital_id=hospital.name,
                        study_id=study.name,
                        study_path=study,
                        meta=meta,
                    )
                )
        return records

    def _build_hospital_index(self) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = {}
        for idx, record in enumerate(self.records):
            mapping.setdefault(record.hospital_id, []).append(idx)
        return mapping


class ToyStudyDataset(Dataset[dict[str, Any]]):
    """Synthetic dataset with domain styles and artifacts for tests."""

    def __init__(
        self,
        num_hospitals: int = 3,
        studies_per_hospital: int = 4,
        slices_per_study: int = 8,
        image_size: int = 64,
        seed: int = 7,
    ) -> None:
        self.num_hospitals = num_hospitals
        self.studies_per_hospital = studies_per_hospital
        self.slices_per_study = slices_per_study
        self.image_size = image_size
        self.seed = seed
        self.records = self._build_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        rng = np.random.default_rng(self.seed + idx)
        base = rng.normal(
            0.5, 0.15, size=(self.slices_per_study, self.image_size, self.image_size)
        )
        base = np.clip(base, 0.0, 1.0)
        base = _apply_domain_style(base, record["hospital_id"], self.seed)
        # Embed a weak label signal for learnability
        base = np.clip(base + 0.2 * float(record["label"]), 0.0, 1.0)
        base = _apply_artifacts(base, rng)
        slices = torch.from_numpy(base[:, None, :, :].astype(np.float32))
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return {
            "hospital_id": record["hospital_id"],
            "study_id": record["study_id"],
            "slices": slices,
            "label": label,
            "meta": record["meta"],
        }

    def _build_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for h in range(self.num_hospitals):
            hospital_id = f"hospital_{h:03d}"
            for s in range(self.studies_per_hospital):
                study_id = f"study_{h:03d}_{s:03d}"
                label = (s + h) % 2
                meta = {
                    "modality": "CT" if h % 2 == 0 else "MRI",
                    "body_part": "head",
                    "label": label,
                    "scanner_id": f"scanner_{h:02d}",
                }
                records.append(
                    {
                        "hospital_id": hospital_id,
                        "study_id": study_id,
                        "label": label,
                        "meta": meta,
                    }
                )
        return records


def collate_mil(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length slice stacks and create attention mask."""
    batch_size = len(batch)
    max_slices = max(item["slices"].shape[0] for item in batch)
    c, h, w = batch[0]["slices"].shape[1:]
    padded = torch.zeros(
        (batch_size, max_slices, c, h, w), dtype=batch[0]["slices"].dtype
    )
    mask = torch.zeros((batch_size, max_slices), dtype=torch.bool)
    for i, item in enumerate(batch):
        s = item["slices"].shape[0]
        padded[i, :s] = item["slices"]
        mask[i, :s] = True
    return {
        "hospital_id": [item["hospital_id"] for item in batch],
        "study_id": [item["study_id"] for item in batch],
        "slices": padded,
        "attention_mask": mask,
        "label": torch.stack([item["label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


def _list_slices(study_path: Path) -> list[Path]:
    slice_paths = sorted(
        [p for p in study_path.iterdir() if p.suffix.lower() in {".npy", ".png"}],
        key=lambda p: p.name,
    )
    if not slice_paths:
        raise FileNotFoundError(f"No slices found in {study_path}")
    return slice_paths


def _sample_slices(
    paths: list[Path],
    study_id: str,
    max_slices: int | None,
    strategy: str,
    seed: int,
) -> list[Path]:
    if max_slices is None or len(paths) <= max_slices:
        return paths
    if strategy == "center":
        start = max(0, (len(paths) - max_slices) // 2)
        return paths[start : start + max_slices]
    if strategy == "uniform":
        idx = np.linspace(0, len(paths) - 1, max_slices).astype(int)
        return [paths[i] for i in idx]
    if strategy == "random":
        rng = np.random.default_rng(_stable_seed(study_id, seed))
        idx = rng.choice(len(paths), size=max_slices, replace=False)
        return [paths[i] for i in sorted(idx.tolist())]
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _stable_seed(study_id: str, seed: int) -> int:
    h = hashlib.md5(study_id.encode("utf-8")).hexdigest()[:8]
    return seed + int(h, 16)


def _load_slice(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def _apply_preprocess(
    img: np.ndarray, meta: dict[str, Any], cfg: PreprocessConfig
) -> np.ndarray:
    modality = str(meta.get("modality", "")).upper()
    if (
        modality == "CT"
        and cfg.ct_window_level is not None
        and cfg.ct_window_width is not None
    ):
        low = cfg.ct_window_level - cfg.ct_window_width / 2
        high = cfg.ct_window_level + cfg.ct_window_width / 2
        img = np.clip(img, low, high)
        img = (img - low) / max(high - low, 1e-6)
    elif modality == "MRI" and cfg.mri_norm:
        mean = float(img.mean())
        std = float(img.std())
        img = (img - mean) / max(std, 1e-6)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    else:
        if img.max() > 1.0:
            img = img / 255.0
    return img.astype(np.float32)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[None, :, :]
    return torch.from_numpy(img.astype(np.float32))


def _apply_domain_style(volume: np.ndarray, hospital_id: str, seed: int) -> np.ndarray:
    # Simple domain styles based on hospital index
    h_idx = int(hospital_id.split("_")[-1]) if "_" in hospital_id else 0
    rng = np.random.default_rng(seed + h_idx)
    contrast = 0.8 + 0.4 * rng.random()
    bias = -0.1 + 0.2 * rng.random()
    styled = np.clip(volume * contrast + bias, 0.0, 1.0)
    return styled


def _apply_artifacts(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Add synthetic streak artifacts
    v = volume.copy()
    if rng.random() < 0.5:
        for s in range(v.shape[0]):
            x = rng.integers(0, v.shape[2])
            v[s, :, x : x + 2] = np.clip(v[s, :, x : x + 2] + 0.4, 0.0, 1.0)
    if rng.random() < 0.3:
        for s in range(v.shape[0]):
            y = rng.integers(0, v.shape[1])
            v[s, y : y + 2, :] = np.clip(v[s, y : y + 2, :] - 0.3, 0.0, 1.0)
    return v
