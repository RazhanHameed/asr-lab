"""ASR model architectures.

Available models:
- SSM: State Space Model (Mamba2) based ASR
- OmniRouter: Omni-Router MoE ASR with shared routing (Apple ASRU 2025)
- Whisper: Encoder-decoder transformer ASR
- Conformer: Fast Conformer with CTC/RNNT
"""

from asr_lab.models.base import (
    ASRModel,
    CTCDecoder,
    Decoder,
    Encoder,
    ModelConfig,
    TransducerDecoder,
)
from asr_lab.models.omni_router import OmniRouterASRModel, OmniRouterConfig
from asr_lab.models.ssm import SSMASRModel, SSMConfig

__all__ = [
    "ASRModel",
    "CTCDecoder",
    "Decoder",
    "Encoder",
    "ModelConfig",
    "OmniRouterASRModel",
    "OmniRouterConfig",
    "SSMASRModel",
    "SSMConfig",
    "TransducerDecoder",
]
