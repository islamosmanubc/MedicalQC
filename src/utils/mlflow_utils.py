"""MLflow utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import mlflow

from src.utils.env import collect_env_info
from src.utils.io import ensure_dir, save_json


def start_run(tracking_uri: str, experiment_name: str, run_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)


def end_run() -> None:
    mlflow.end_run()


def log_params_recursive(params: dict[str, Any]) -> None:
    flat = _flatten_dict(params)
    mlflow.log_params(flat)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(paths: Iterable[Path], artifact_path: str | None = None) -> None:
    for path in paths:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_config_artifact(cfg_resolved: dict[str, Any], output_dir: Path) -> Path:
    ensure_dir(output_dir)
    config_path = output_dir / "resolved_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_resolved, f, indent=2, sort_keys=True)
    mlflow.log_artifact(str(config_path), artifact_path="repro")
    return config_path


def log_git_commit(output_dir: Path) -> Path:
    ensure_dir(output_dir)
    commit_path = output_dir / "git_commit.txt"
    commit = "unknown"
    with suppress(Exception):
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
    commit_path.write_text(commit + "\n", encoding="utf-8")
    mlflow.log_artifact(str(commit_path), artifact_path="repro")
    return commit_path


def log_pip_freeze(output_dir: Path) -> Path:
    ensure_dir(output_dir)
    freeze_path = output_dir / "pip_freeze.txt"
    text = "pip freeze unavailable"
    with suppress(Exception):
        text = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT
        ).decode("utf-8")
    freeze_path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(freeze_path), artifact_path="repro")
    return freeze_path


def log_system_info(output_dir: Path) -> Path:
    ensure_dir(output_dir)
    info_path = output_dir / "system_info.json"
    info = collect_env_info()
    save_json(info, info_path)
    mlflow.log_artifact(str(info_path), artifact_path="repro")
    return info_path


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, name))
        else:
            flat[name] = value
    return flat
