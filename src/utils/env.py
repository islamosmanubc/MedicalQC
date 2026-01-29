"""Environment collection."""

from __future__ import annotations

import os
import platform
from typing import Any

import torch


def collect_env_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
