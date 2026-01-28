"""Hydra config registration."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from src.configs.schema import AppConfig, DataConfig, EvalConfig, FederatedConfig, LoggingConfig, MlflowConfig, ModelConfig, TrainConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=AppConfig)
    cs.store(group="data", name="data", node=DataConfig)
    cs.store(group="model", name="model", node=ModelConfig)
    cs.store(group="train", name="train", node=TrainConfig)
    cs.store(group="federated", name="federated", node=FederatedConfig)
    cs.store(group="logging", name="logging", node=LoggingConfig)
    cs.store(group="mlflow", name="mlflow", node=MlflowConfig)
    cs.store(group="eval", name="eval", node=EvalConfig)
