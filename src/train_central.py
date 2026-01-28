"""Central training entrypoint using Hydra and MLflow (canonical src)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from src.configs.schema import AppConfig
from src.data.datamodule import DataConfig as DMDataConfig, LoaderConfig, build_dataset
from src.data.datasets import PreprocessConfig, SamplingConfig, collate_mil
from src.models.qc_model import QCFederatedMILModel
from src.eval.evaluator import EvalConfig as EvalRunConfig
from src.train.central_trainer import CentralConfig, train_central
from src.utils.config import init_hydra, log_config, resolve_config, setup_runtime, validate_config
from src.utils.io import ensure_dir
from src.utils.logging import configure_logging, get_logger
from src.utils.mlflow_utils import (
    end_run,
    log_config_artifact,
    log_git_commit,
    log_metrics,
    log_params_recursive,
    log_pip_freeze,
    log_system_info,
    start_run,
)

init_hydra()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app_cfg: AppConfig = cfg  # type: ignore[assignment]
    configure_logging(app_cfg.logging.level)
    logger = get_logger("train_central")

    validate_config(app_cfg)
    setup_runtime(app_cfg)
    log_config(logger, app_cfg)

    device = torch.device(app_cfg.train.device)

    data_cfg = DMDataConfig(
        root=app_cfg.data.dataset_root,
        backend="toy" if app_cfg.data.dataset_type == "toy" else "folder",
        preprocess=PreprocessConfig(
            ct_window_level=app_cfg.data.ct_window_level,
            ct_window_width=app_cfg.data.ct_window_width,
            mri_norm=app_cfg.data.mri_norm,
        ),
        sampling=SamplingConfig(
            max_slices=app_cfg.data.sampling_max_slices,
            strategy=app_cfg.data.sampling_strategy,
        ),
        seed=app_cfg.data.seed,
        num_hospitals=app_cfg.data.num_hospitals,
        studies_per_hospital=app_cfg.data.studies_per_hospital,
        slices_per_study=app_cfg.data.slices_per_study,
        image_size=app_cfg.data.image_size,
    )
    loader_cfg = LoaderConfig(
        batch_size=app_cfg.train.batch_size,
        num_workers=app_cfg.data.num_workers,
        pin_memory=app_cfg.data.pin_memory,
        seed=app_cfg.data.seed,
    )

    dataset = build_dataset(data_cfg)
    hospital_map = _build_hospital_map(dataset)
    holdout_ids = _resolve_holdout_ids(app_cfg.federated.holdout_clients, list(hospital_map.keys()))

    train_indices = [i for h, idxs in hospital_map.items() if h not in holdout_ids for i in idxs]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        collate_fn=collate_mil,
    )

    holdout_loaders: Dict[str, DataLoader] = {}
    for hid in holdout_ids:
        holdout_loaders[hid] = DataLoader(
            Subset(dataset, hospital_map[hid]),
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            collate_fn=collate_mil,
        )

    model = QCFederatedMILModel(app_cfg.model.to_qc_config()).to(device)

    start_run(app_cfg.mlflow.tracking_uri, app_cfg.mlflow.experiment_name, app_cfg.mlflow.run_name)
    output_dir = ensure_dir(Path(app_cfg.output_dir) / "artifacts")
    resolved = resolve_config(app_cfg)
    log_params_recursive(resolved)
    log_config_artifact(resolved, output_dir)
    log_git_commit(output_dir)
    log_pip_freeze(output_dir)
    log_system_info(output_dir)

    history = train_central(
        model,
        train_loader,
        holdout_loaders,
        CentralConfig(
            epochs=app_cfg.train.epochs,
            lr=app_cfg.train.lr,
            weight_decay=app_cfg.train.weight_decay,
            mixed_precision=app_cfg.train.mixed_precision,
            grad_clip_norm=app_cfg.train.grad_clip_norm,
            device=app_cfg.train.device,
            output_dir=app_cfg.output_dir,
            eval_cfg=EvalRunConfig(
                specificity_target=app_cfg.eval.specificity_target,
                sensitivity_target=app_cfg.eval.sensitivity_target,
                output_dir=app_cfg.output_dir,
            ),
        ),
    )

    if history:
        log_metrics({"final_train_loss": history[-1]["train_loss"]})
    end_run()


def _build_hospital_map(dataset) -> Dict[str, List[int]]:
    if hasattr(dataset, "hospital_to_indices"):
        return dataset.hospital_to_indices
    mapping: Dict[str, List[int]] = {}
    if hasattr(dataset, "records"):
        for idx, rec in enumerate(dataset.records):
            hid = rec["hospital_id"] if isinstance(rec, dict) else rec.hospital_id
            mapping.setdefault(hid, []).append(idx)
    return mapping


def _resolve_holdout_ids(holdout, hospital_ids: List[str]) -> List[str]:
    resolved: List[str] = []
    for item in holdout:
        if isinstance(item, int):
            resolved.append(f"hospital_{item:03d}")
        else:
            resolved.append(str(item))
    # filter to existing
    return [h for h in resolved if h in set(hospital_ids)]


if __name__ == "__main__":
    main()
