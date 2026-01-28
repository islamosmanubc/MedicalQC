import json
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import PreprocessConfig, SamplingConfig, StudyFolderDataset, collate_mil


def _write_study(root: Path, hospital: str, study: str, num_slices: int) -> None:
    study_dir = root / hospital / study
    study_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "modality": "CT",
        "body_part": "head",
        "label": 1,
        "scanner_id": "scanner_x",
    }
    (study_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for i in range(num_slices):
        arr = (np.ones((8, 8)) * i).astype(np.float32)
        np.save(study_dir / f"slice_{i:03d}.npy", arr)


def test_collate_padding_and_mask(tmp_path: Path):
    _write_study(tmp_path, "hospital_000", "study_000", 3)
    _write_study(tmp_path, "hospital_000", "study_001", 5)

    dataset = StudyFolderDataset(
        root=tmp_path,
        preprocess=PreprocessConfig(ct_window_level=40, ct_window_width=80),
        sampling=SamplingConfig(max_slices=None),
        seed=0,
    )
    batch = [dataset[0], dataset[1]]
    out = collate_mil(batch)

    assert out["slices"].shape == (2, 5, 1, 8, 8)
    assert out["attention_mask"].shape == (2, 5)
    assert out["attention_mask"][0].sum().item() == 3
    assert out["attention_mask"][1].sum().item() == 5
    assert out["label"].shape == torch.Size([2])
