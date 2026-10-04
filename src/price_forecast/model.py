"""Loading and calling the TimesFM 3.0 forecaster."""

from __future__ import annotations

import os
import time
from pathlib import Path

# Keep the ~1.2 GB checkpoint in the project (git-ignored) instead of the global HF cache.
# Must be set before huggingface_hub is imported; an explicit HF_HUB_CACHE still wins.
os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / "models" / "hf"))

import numpy as np
import torch
from timesfm3 import ForecastOutput, ModelConfig, TimesFM3Evaluator

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
CHECKPOINT = "google/timesfm-3.0-pytorch"


def resolve_device(name: str) -> str:
    """Turn ``auto`` into the best device actually available on this machine."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_forecaster(device: str, batch_size: int = 8) -> TimesFM3Evaluator:
    config = ModelConfig(
        checkpoint_path=CHECKPOINT,
        per_core_batch_size=batch_size,
        device=device,
    )
    start = time.perf_counter()
    forecaster = TimesFM3Evaluator(config)
    print(f"  loaded checkpoint on {device} in {time.perf_counter() - start:.1f}s")
    return forecaster


def predict(
    forecaster: TimesFM3Evaluator,
    contexts: list[np.ndarray],
    horizon: int,
    **kwargs,
) -> tuple[list[ForecastOutput], float]:
    """Run ``predict_batch`` and report wall-clock time."""
    start = time.perf_counter()
    outputs = list(
        forecaster.predict_batch(
            contexts=contexts,
            horizon=horizon,
            **{"return_quantiles": True, "use_symmetric_averaging": False, **kwargs},
        )
    )
    return outputs, time.perf_counter() - start
