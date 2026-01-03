"""ASR Lab: Modular ASR Training Framework.

A unified framework for training and evaluating ASR models with different architectures:
- SSM (State Space Models) with Mamba2
- Whisper-like encoder-decoder
- Fast Conformer

Features:
- Modular encoder/decoder architecture
- Reproducible training with RepDL integration
- Multi-precision support (FP32, FP16, BF16, FP8, MXFP8, MXFP4)
- Streaming and offline inference modes
- Hydra-based configuration
"""

__version__ = "0.1.0"

from asr_lab.models.base import (
    ASRModel,
    Encoder,
    Decoder,
    CTCDecoder,
    TransducerDecoder,
)
from asr_lab.audio.features import (
    FeatureExtractor,
    MelSpectrogramExtractor,
    MFCCExtractor,
)
from asr_lab.audio.augmentation import SpecAugment
from asr_lab.tokenizers.base import Tokenizer, CharacterTokenizer, BPETokenizer
from asr_lab.training.precision import (
    PrecisionMode,
    PrecisionManager,
    get_precision_manager,
)
from asr_lab.training.trainer import Trainer

__all__ = [
    # Models
    "ASRModel",
    "Encoder",
    "Decoder",
    "CTCDecoder",
    "TransducerDecoder",
    # Audio
    "FeatureExtractor",
    "MelSpectrogramExtractor",
    "MFCCExtractor",
    "SpecAugment",
    # Tokenizers
    "Tokenizer",
    "CharacterTokenizer",
    "BPETokenizer",
    # Training
    "PrecisionMode",
    "PrecisionManager",
    "get_precision_manager",
    "Trainer",
]
