"""Audio processing utilities for ASR.

This module provides:
- Feature extraction (Mel spectrogram, MFCC)
- Data augmentation (SpecAugment, noise injection)
- Audio loading and preprocessing
"""

from asr_lab.audio.features import (
    FeatureExtractor,
    MelSpectrogramExtractor,
    MFCCExtractor,
)
from asr_lab.audio.augmentation import SpecAugment

__all__ = [
    "FeatureExtractor",
    "MelSpectrogramExtractor",
    "MFCCExtractor",
    "SpecAugment",
]
