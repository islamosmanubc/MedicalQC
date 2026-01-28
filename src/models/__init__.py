"""Model components for shared encoder, LoRA, and spectral fusion."""

from src.models.lora import LoraConfig, inject_lora
from src.models.mil import AttentionMILPool, HybridAggregatorConfig, HybridStudyAggregator, WorstSliceTopK
from src.models.model import SpectralMILModel
from src.models.qc_model import QCModelConfig, QCFederatedMILModel
from src.models.spectral import Fusion, SpectralConfig, SpectralEncoder
from src.models.swin_encoder import SwinEncoder
from src.models.uncertainty import (
    EvidentialBinaryHead,
    EvidentialConfig,
    TemperatureScaler,
    decision_policy,
    expected_calibration_error,
    mc_dropout_predict,
    reliability_diagram,
)

__all__ = [
    "LoraConfig",
    "inject_lora",
    "SwinEncoder",
    "SpectralConfig",
    "SpectralEncoder",
    "Fusion",
    "SpectralMILModel",
    "AttentionMILPool",
    "WorstSliceTopK",
    "HybridAggregatorConfig",
    "HybridStudyAggregator",
    "QCModelConfig",
    "QCFederatedMILModel",
    "EvidentialBinaryHead",
    "EvidentialConfig",
    "TemperatureScaler",
    "mc_dropout_predict",
    "expected_calibration_error",
    "reliability_diagram",
    "decision_policy",
]
