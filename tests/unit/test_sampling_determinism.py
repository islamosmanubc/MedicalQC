import json
from pathlib import Path

import numpy as np

from src.data.datasets import SamplingConfig, StudyFolderDataset


def _write_study(root: Path, hospital: str, study: str, num_slices: int) -> None:
    study_dir = root / hospital / study
    study_dir.mkdir(parents=True, exist_ok=True)
    meta = {"modality": "CT", "body_part": "head", "label": 0}
    (study_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for i in range(num_slices):
        arr = (np.ones((8, 8)) * i).astype(np.float32)
        np.save(study_dir / f"slice_{i:03d}.npy", arr)


def test_uniform_sampling_deterministic(tmp_path: Path):
    _write_study(tmp_path, "hospital_000", "study_000", 10)
    ds1 = StudyFolderDataset(
        root=tmp_path,
        sampling=SamplingConfig(max_slices=4, strategy="uniform"),
        seed=123,
    )
    ds2 = StudyFolderDataset(
        root=tmp_path,
        sampling=SamplingConfig(max_slices=4, strategy="uniform"),
        seed=999,
    )

    s1 = ds1[0]["slices"]
    s2 = ds2[0]["slices"]

    assert np.allclose(s1.numpy(), s2.numpy())


def test_random_sampling_deterministic(tmp_path: Path):
    _write_study(tmp_path, "hospital_000", "study_000", 10)
    ds1 = StudyFolderDataset(
        root=tmp_path,
        sampling=SamplingConfig(max_slices=4, strategy="random"),
        seed=123,
    )
    ds2 = StudyFolderDataset(
        root=tmp_path,
        sampling=SamplingConfig(max_slices=4, strategy="random"),
        seed=123,
    )

    s1 = ds1[0]["slices"]
    s2 = ds2[0]["slices"]

    assert np.allclose(s1.numpy(), s2.numpy())
