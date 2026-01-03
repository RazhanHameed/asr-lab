"""Whisper-like encoder-decoder ASR model.

This module implements a Whisper-style ASR with:
- Convolutional audio encoder
- Transformer encoder
- Transformer decoder with cross-attention
"""

from asr_lab.models.whisper.config import WhisperConfig
from asr_lab.models.whisper.encoder import WhisperEncoder
from asr_lab.models.whisper.decoder import WhisperDecoder
from asr_lab.models.whisper.model import WhisperASRModel

__all__ = [
    "WhisperConfig",
    "WhisperEncoder",
    "WhisperDecoder",
    "WhisperASRModel",
]
