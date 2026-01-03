"""Training utilities for ASR models.

This module provides:
- Precision management (FP32, BF16, FP8, MXFP8)
- Trainer class with distributed training support
- Dataset utilities
"""

from asr_lab.training.precision import (
    PrecisionMode,
    PrecisionConfig,
    PrecisionManager,
    get_precision_manager,
    detect_optimal_precision,
)
from asr_lab.training.trainer import Trainer, TrainerConfig
from asr_lab.training.dataset import ASRDataset, ASRCollator

__all__ = [
    "PrecisionMode",
    "PrecisionConfig",
    "PrecisionManager",
    "get_precision_manager",
    "detect_optimal_precision",
    "Trainer",
    "TrainerConfig",
    "ASRDataset",
    "ASRCollator",
]
