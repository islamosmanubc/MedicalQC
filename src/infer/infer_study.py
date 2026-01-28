"""Study inference pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.datasets import PreprocessConfig, SamplingConfig, StudyFolderDataset, collate_mil
from src.models.qc_model import QCModelConfig, QCFederatedMILModel
from src.models.spectral import SpectralConfig
from src.utils.logging import configure_logging, get_logger


def infer_study(
    study_path: Path,
    model_path: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    logger = get_logger("infer")
    if not study_path.exists():
        raise FileNotFoundError(f"Study path not found: {study_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Minimal config for inference
    cfg = QCModelConfig(
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        in_channels=1,
        embed_dim=64,
        spectral=SpectralConfig(out_dim=16, mode="radial", target_size=32),
        fusion_mode="concat_mlp",
        fusion_dim=32,
        attn_hidden=32,
        dropout=0.0,
        uncertainty_mode="evidential",
    )
    model = QCFederatedMILModel(cfg)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        model.load_shared_state_dict(state)
    model.to(torch.device(device))
    model.eval()

    # build single-study dataset
    dataset = StudyFolderDataset(
        root=study_path.parent.parent,
        preprocess=PreprocessConfig(),
        sampling=SamplingConfig(),
        seed=0,
    )
    # locate study index
    idx = None
    for i, rec in enumerate(dataset.records):
        if rec.study_path == study_path:
            idx = i
            break
    if idx is None:
        raise RuntimeError("Study not found in dataset index")
    batch = collate_mil([dataset[idx]])
    _validate_batch(batch)

    with torch.no_grad():
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(batch)

    p_fail = float(outputs["p_fail"].item())
    u = float(outputs["u"].item())
    decision = _decision(p_fail, u)
    attention = outputs["attention_weights"].detach().cpu().numpy()
    attn_summary = {
        "mean": float(attention.mean()),
        "max": float(attention.max()),
        "min": float(attention.min()),
    }

    report = {
        "decision": decision,
        "p_fail": p_fail,
        "u": u,
        "attention_summary": attn_summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": "0.1.0",
    }
    logger.info("Inference report: %s", report)
    return report


def _decision(p_fail: float, u: float, t_fail: float = 0.5, t_u: float = 0.4) -> str:
    if u >= t_u:
        return "UNCERTAIN"
    if p_fail >= t_fail:
        return "FAIL"
    return "PASS"


def _validate_batch(batch: Dict[str, Any]) -> None:
    if "slices" not in batch:
        raise ValueError("Batch missing 'slices'")
    x = batch["slices"]
    if not torch.is_tensor(x):
        raise ValueError("'slices' must be a tensor")
    if x.dim() != 5:
        raise ValueError("'slices' must have shape [B,S,C,H,W]")
    if torch.isnan(x).any():
        raise ValueError("'slices' contains NaNs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    configure_logging("INFO")
    report = infer_study(Path(args.study), Path(args.model), args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
