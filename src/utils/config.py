"""Hydra config utilities: validate, seed, determinism."""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import OmegaConf

from src.configs.schema import AppConfig
from src.configs.hydra import register_configs
from src.utils.seed import set_global_seed


def init_hydra() -> None:
    """Register structured configs with Hydra."""
    register_configs()


def validate_config(cfg: AppConfig) -> None:
    """Lightweight validation of key config fields."""
    if cfg.train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0")
    if cfg.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0")
    if cfg.data.image_size <= 0:
        raise ValueError("data.image_size must be > 0")
    if cfg.model.dropout < 0.0:
        raise ValueError("model.dropout must be >= 0")
    if cfg.data.slices_per_study <= 0:
        raise ValueError("data.slices_per_study must be > 0")


def setup_runtime(cfg: AppConfig) -> None:
    """Set seeds and deterministic flags."""
    set_global_seed(cfg.train.seed, cfg.train.deterministic)
    if cfg.train.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def log_config(logger: logging.Logger, cfg: AppConfig) -> None:
    """Log resolved config."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))


def resolve_config(cfg: Any) -> Any:
    """Ensure config is resolved for logging and MLflow params."""
    return OmegaConf.to_container(cfg, resolve=True)
