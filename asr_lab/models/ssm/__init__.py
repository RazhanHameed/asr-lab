"""SSM (State Space Model) based ASR with Mamba2.

This module implements an SSM-based ASR encoder using Mamba2 blocks
with optional Flash Attention layers.
"""

from asr_lab.models.ssm.config import SSMConfig
from asr_lab.models.ssm.encoder import SSMEncoder
from asr_lab.models.ssm.model import SSMASRModel

__all__ = [
    "SSMConfig",
    "SSMEncoder",
    "SSMASRModel",
]
